import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY


# -------------------------- 维度变换工具函数 适配LayerNorm与频域操作 --------------------------
def to_3d(x):
    """将4D图像特征[B,C,H,W]转为3D序列特征[B,N,C]，适配LayerNorm计算"""
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    """将3D序列特征还原为4D图像特征，适配卷积与频域操作"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# -------------------------- 适配图像的LayerNorm模块 对应结构图LayerNorm模块 --------------------------
class BiasFree_LayerNorm(nn.Module):
    """无偏置层归一化，适配频域特征的通道级归一化"""
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    """带偏置层归一化，Transformer标准归一化实现"""
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    """图像特征统一LayerNorm封装，支持带偏置/无偏置两种模式，适配4D图像输入"""
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class DualBranchFeatureDecouple(nn.Module):
    """
    特征双分支解耦模块（已修复 FLOPs 统计报错）
    输入：x -> (B, C, H, W)
    输出：low_feat, high_feat  均为 (B, C, H, W)
    解耦：低频结构特征 ｜ 高频细节特征
    支持 fvcore FlopCountAnalysis 不报错
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim

        # 输入映射
        self.conv_in = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)

        # 低频分支：平滑 + 语义（替换掉了报错的 AdaptiveAvgPool2d(None)）
        self.low_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),  # 固定尺寸，不报错
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),  # 用 BN 替代固定尺寸 LayerNorm，兼容任意 H,W
            nn.GELU()
        )

        # 高频分支：边缘 + 纹理
        self.high_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # 通道门控
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim * 2, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_in(x)

        # 低频
        low_raw = self.low_branch(x)

        # 高频 = 原图 - 低频
        high_raw = x - low_raw
        high_raw = self.high_branch(high_raw)

        # 门控
        attn = self.attention(x)
        gate_low, gate_high = torch.chunk(attn, 2, dim=1)

        low_feat = low_raw * gate_low
        high_feat = high_raw * gate_high

        return low_feat, high_feat

class FDCA(nn.Module):
    """
    FDCA: Multimodal Frequency Domain Cross-Attention 多模态频域交叉注意力
    完整对应结构图左侧全流程，核心功能：在频域实现红外与可见光的双向全局交互与互补增强
    """
    def __init__(self, dim, bias):
        super(FDCA, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        # QKV生成：1×1卷积+深度可分离卷积
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')
        self.patch_size = 2

    def forward(self, x, y):
        # 输入双模态特征：rgb_fea=可见光特征E^G_RGB，ir_fea=红外特征E^G_IR
        rgb_fea = x
        ir_fea = y
        # 步骤1：层归一化 对应结构图LayerNorm模块
        rgb_fea_norm = self.norm1(rgb_fea)
        ir_fea_norm = self.norm1(ir_fea)
        # 步骤2：QKV特征映射 对应结构图Conv+Dw-Conv模块
        rgb_hidden = self.to_hidden(rgb_fea_norm)
        ir_hidden = self.to_hidden(ir_fea_norm)
        # 拆分Q/K/V，每个模态各生成一组QKV
        rgb_q, rgb_k, rgb_v = self.to_hidden_dw(rgb_hidden).chunk(3, dim=1)
        ir_q, ir_k, ir_v = self.to_hidden_dw(ir_hidden).chunk(3, dim=1)
        # 步骤3：FFT变换到频域 对应结构图FFT模块
        rgb_q_fft = torch.fft.rfft2(rgb_q.float())
        rgb_k_fft = torch.fft.rfft2(rgb_k.float())
        ir_q_fft = torch.fft.rfft2(ir_q.float())
        ir_k_fft = torch.fft.rfft2(ir_k.float())
        # 步骤4：频域注意力计算 对应结构图⊙元素乘模块
        rgb_out = rgb_q_fft * rgb_k_fft  # 可见光模态频域自注意力
        ir_out = ir_q_fft * ir_k_fft    # 红外模态频域自注意力
        # 步骤5：IFFT逆变换回空域 对应结构图IFFT模块
        rgb_out = torch.fft.irfft2(rgb_out, s=(rgb_q.shape[2], rgb_q.shape[3]))
        ir_out = torch.fft.irfft2(ir_out, s=(ir_q.shape[2], ir_q.shape[3]))
        # 步骤6：层归一化
        rgb_out = self.norm(rgb_out)
        ir_out = self.norm(ir_out)
        # 步骤7：跨模态双向加权 对应结构图跨模态V加权环节
        rgb_output = ir_v * rgb_out  # 红外V加权可见光特征，补充结构信息
        ir_output = rgb_v * ir_out  # 可见光V加权红外特征，补充细节信息
        # 步骤8：输出投影
        rgb_output = self.project_out(rgb_output)
        ir_output = self.project_out(ir_output)
        return rgb_output, ir_output

# -------------------------- FDFAM核心模块 完整对应结构图全流程 --------------------------
class FDFTM(nn.Module):
    """
    FDFAM: Frequency Domain Feature Aggregation Module 频域特征聚合模块（代码中命名为FDFTM）
    完整对应结构图全流程，核心功能：频域跨模态注意力+多尺度频域前馈网络，实现红外-可见光双模态特征的精准聚合与增强
    输入：[可见光特征, 红外特征]，输出：融合后的聚合特征
    """
    def __init__(self, dim):
        super(FDFTM, self).__init__()
        bias = False
        LayerNorm_type = 'WithBias'
        ffn_expansion_factor = 3
        # 多模态频域交叉注意力模块
        self.attn = FDCA(dim, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # 多尺度频域前馈网络
        # self.ffn = FDFFN(dim, ffn_expansion_factor, bias)
        self.ffn = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        # 特征拼接与融合
        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # 输入双模态特征
        rgb_fea = x
        ir_fea = y
        # 步骤1：频域跨模态注意力增强
        rgb_fea_out, ir_fea_out = self.attn(rgb_fea, ir_fea)
        # 注意力残差连接
        rgb_att_out = rgb_fea + rgb_fea_out
        ir_att_out = ir_fea + ir_fea_out
        # 步骤2：频域前馈网络特征增强
        rgb_fea = rgb_att_out + self.ffn(self.norm2(rgb_att_out))
        ir_fea = ir_att_out + self.ffn(self.norm2(ir_att_out))
        # 步骤3：双模态特征拼接与融合
        fea_cat = torch.cat([rgb_fea, ir_fea], dim=1)
        out_fea = self.relu(self.conv(fea_cat))
        return out_fea

class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim,hidden_dim,1,1,0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim ,3,1,1)

        self.act =nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x,[self.p_dim,self.hidden_dim-self.p_dim],dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1,x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:,:self.p_dim,:,:] = self.act(self.conv_1(x[:,:self.p_dim,:,:]))
            x = self.conv_2(x)
        return x

class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim,dim*2,1,1,0)
        self.linear_1 = nn.Conv2d(dim,dim,1,1,0)
        self.linear_2 = nn.Conv2d(dim,dim,1,1,0)

        self.lde = DMlp(dim,2)

        self.dw_conv = nn.Conv2d(dim,dim,3,1,1,groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1,dim,1,1)))
        self.belt = nn.Parameter(torch.zeros((1,dim,1,1)))

    def forward(self, f):
        _,_,h,w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2,-1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h,w), mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)

class FMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.smfa = SMFA(dim)
        self.pcfn = PCFN(dim, ffn_scale)

    def forward(self, x):
        x = self.smfa(F.normalize(x)) + x
        x = self.pcfn(F.normalize(x)) + x
        return x

# @ARCH_REGISTRY.register()
class FCMANet(nn.Module):
    def __init__(self, dim=36, n_blocks=8, upscaling_factor=4):
        super().__init__()
        # self.fsd = FrequencyStructuralDecoupling(dim)
        self.fd = DualBranchFeatureDecouple(dim)
        self.feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feat1 = nn.Sequential(*[FMB(dim) for _ in range(n_blocks)])
        self.feat2 = nn.Sequential(*[FMB(dim) for _ in range(n_blocks)])
        self.c = FDFTM(dim)

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, ir):
        ir = self.feat(ir)
        l, h = self.fd(ir)
        el = self.feat1(l)
        eh = self.feat2(h)
        e = self.c(el, eh) + ir
        x = self.to_img(e)
        return x

# 模块测试代码
if __name__ == "__main__":
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # x = torch.randn(1, 3, 640, 480)
    x = torch.randn(1, 3, 320, 240)

    model = FCMANet(dim=36, n_blocks=8, upscaling_factor=4)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)