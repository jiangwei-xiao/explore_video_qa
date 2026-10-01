"""从冻结归一化post-LayerNorm CLS恢复BLIP官方视觉投影；不可用于任意模型特征。"""
import numpy as np


def load_projection_parameters(model_file):
    """接口：只读本地BLIP权重，提取末端LayerNorm和官方ITC投影，不加载GPU模型。"""
    import torch
    state=torch.load(model_file,map_location='cpu',weights_only=True)
    keys=dict(gamma='vision_model.post_layernorm.weight',beta='vision_model.post_layernorm.bias',
        weight='vision_proj.weight',bias='vision_proj.bias')
    return {name:state[key].float().numpy().astype(np.float64) for name,key in keys.items()}


def recover_projected_features(features,parameters):
    """接口：仅接受L2归一化的BLIP末端CLS；返回官方投影归一化向量与恢复尺度。"""
    f=np.asarray(features,dtype=np.float64);gamma,beta,W,b=(parameters[k] for k in ('gamma','beta','weight','bias'))
    if f.ndim!=2 or gamma.shape!=beta.shape or f.shape[1]!=len(gamma) or W.shape[1]!=len(gamma) or b.shape!=(W.shape[0],):raise ValueError('Projection shape mismatch')
    if not all(np.isfinite(x).all() for x in (f,gamma,beta,W,b)) or not np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5,rtol=0):raise ValueError('Invalid normalized source')
    if np.any(np.abs(gamma)<1e-12):raise ValueError('Non-invertible LayerNorm scale')
    # 步骤1：x=s*f，x=gamma*z+beta且sum(z)=0，所以s=sum(beta/gamma)/sum(f/gamma)。
    # 不忽略投影偏置；直接把归一化f送入Wf+b通常不等价于官方Wx+b。
    numerator=float(np.sum(beta/gamma));denominator=np.sum(f/gamma,axis=1)
    if abs(numerator)<1e-12 or np.any(np.abs(denominator)<1e-12):raise ValueError('LayerNorm identity cannot recover scale')
    scales=numerator/denominator
    # 步骤2：LayerNorm归一化量范数不超过sqrt(d)，据此检查恢复值的理论上界。
    upper=np.max(np.abs(gamma))*np.sqrt(len(gamma))+np.linalg.norm(beta)
    if np.any(scales<=0) or np.any(scales>upper*(1+1e-5)):raise ValueError('Recovered scale incompatible with LayerNorm')
    raw=f*scales[:,None];projection=raw@W.T+b
    norms=np.linalg.norm(projection,axis=1)
    if not np.isfinite(projection).all() or np.any(norms<=1e-12):raise ValueError('Invalid projected vector')
    # 步骤3：FP64代数恢复减少累计误差，最终输出FP32；须经真实官方前向1e-6核验才准用于实验。
    return (projection/norms[:,None]).astype(np.float32),scales
