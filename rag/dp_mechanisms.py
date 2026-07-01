# rag/dp_mechanisms.py
"""
差分隐私机制封装
从 DPRAG 项目移植的核心 DP 机制
"""

import numpy as np
import torch
from torch.distributions import Gumbel
from typing import List, Optional, Tuple, Dict, Any


class NotFoundLDTop1(Exception):
    """LDGumbelMechanism 未能找到有效 top-1 时抛出"""
    pass


class DPExpenseOverflow(Exception):
    """DP 隐私预算超支时抛出"""
    pass


class LDGumbelMechanism:
    """
    Limit Domain Gumbel 机制
    
    Paper: Durfee, D., & Rogers, R. M. (2019). Practical differentially private 
    top-k selection with pay-what-you-get composition. NeurIPS.
    
    这是一个用于 token 级别差分隐私投票的核心机制。
    在 majority vote 中通过添加 Gumbel 噪声来实现隐私保护。
    
    参数:
        eps: 隐私预算 epsilon
        delta: 隐私失败概率 delta
        target_eps: 总 epsilon 预算上限
        target_delta: 总 delta 预算上限
        k_bar: DP 候选集大小
    """
    
    def __init__(
        self,
        eps: float,
        delta: float,
        k_bar: int = 10,
        target_eps: Optional[float] = None,
        target_delta: Optional[float] = None,
        delta1: Optional[float] = None,
        subsampling_rate: float = 1.0,
        fail_mode: str = 'ld_pate'
    ):
        self.eps = eps
        self.delta = delta
        self.k_bar = k_bar
        self.target_eps = target_eps
        self.target_delta = target_delta
        self.delta1 = delta if delta1 is None else delta1
        self.subsampling_rate = subsampling_rate
        
        self.total_k = 0  # 成功选择的 token 数
        self.total_queries = 0  # 总查询数
        
        self.fail_mode = fail_mode  # 'ld_pate', 'rand', 'stop', 'raise'
    
    def get_top1(self, cnts: torch.Tensor, dim: int, k_bar: Optional[int] = None, sens: float = 2.0) -> int:
        """
        输入 histogram (cnts)，输出 top-1 索引
        
        参数:
            cnts: token 投票计数 tensor
            dim: 词表大小
            k_bar: DP 候选集大小 (默认使用 self.k_bar)
            sens: L0 敏感度 (默认 2.0)
            
        返回:
            选中的 token ID
        """
        self.total_queries += 1
        if k_bar is None:
            k_bar = self.k_bar
            
        assert k_bar <= dim, f"Invalid k_bar: k_bar ({k_bar}) > dim ({dim})"
        
        real_len_cnts = len(cnts)
        sorted_cnts, sorted_idxs = torch.sort(cnts, descending=True)
        gumbel = Gumbel(0., 1. / self.eps)
        
        output_rand_if_fail = False
        
        if k_bar < dim:
            # 计算阈值
            if k_bar + 1 > real_len_cnts:
                h_perp = 0
            else:
                h_perp = sorted_cnts[k_bar]
            
            h_perp = h_perp + 1 + np.log(min(sens, k_bar, dim - k_bar) / self.delta) / self.eps
            v_perp = h_perp + gumbel.sample()
            v_perp = v_perp.item()
            
            if k_bar > real_len_cnts:
                v_perp_0 = torch.max(gumbel.sample((k_bar - real_len_cnts,))).item()
                if v_perp_0 >= v_perp:
                    v_perp = v_perp_0
                    output_rand_if_fail = True
                k_bar = real_len_cnts
        else:
            # k_bar == dim, 退化为 PATE
            if k_bar > real_len_cnts:
                v_perp = torch.max(gumbel.sample((k_bar - real_len_cnts,))).item()
                k_bar = real_len_cnts
                output_rand_if_fail = True
            else:
                v_perp = -np.inf  # 永不失败
        
        # 添加 Gumbel 噪声并选择
        v_cnts = sorted_cnts[:k_bar] + gumbel.sample((k_bar,)).to(sorted_cnts.device)
        v_max_idx = torch.argmax(v_cnts)
        
        if v_cnts[v_max_idx] >= v_perp:
            # 成功
            self.total_k += 1
            self.check_dp_budget()
            return sorted_idxs[v_max_idx].item()
        else:
            # 失败
            if output_rand_if_fail:
                self.total_k += 1
                rand_idx = torch.randint(real_len_cnts, dim, (1,))[0].item()
                self.check_dp_budget()
                return rand_idx
            else:
                self.check_dp_budget()
                raise NotFoundLDTop1()
    
    def get_dp_expense(self, total_queries: Optional[int] = None) -> Tuple[float, float]:
        """
        获取当前累积的 DP 隐私支出
        
        返回:
            (eps, delta) 元组
        """
        if total_queries is None:
            total_queries = self.total_queries
        
        # 使用简单的 compose 计算
        k = self.total_k
        l = total_queries
        
        # 三种 bounds 取最小
        eps_ = [None] * 3
        eps_[0] = k * self.eps
        eps_[1] = k * self.eps * (np.exp(self.eps) - 1) / (np.exp(self.eps) + 1) + self.eps * np.sqrt(
            2. * k * np.log(1 / self.delta1))
        eps_[2] = k * (self.eps ** 2) / 2. + self.eps * np.sqrt(0.5 * k * np.log(1 / self.delta1))
        
        eps = np.min(eps_)
        delta = 2 * l * self.delta + self.delta1
        
        return eps, delta
    
    def check_dp_budget(self, raise_error: bool = True, verbose: bool = False) -> bool:
        """
        检查隐私预算是否超支
        
        参数:
            raise_error: 超支时是否抛出异常
            verbose: 是否打印详细信息
            
        返回:
            是否仍在预算内
        """
        if self.target_eps is not None:
            eps, delta = self.get_dp_expense()
            if verbose:
                print(f"# dp eps={eps:.4f}, delta={delta:g}")
            
            if eps > self.target_eps or delta > self.target_delta:
                if raise_error:
                    raise DPExpenseOverflow()
                return False
        return True


def majority_vote(
    tokens: torch.Tensor,
    dim: int,
    dp_engine: Optional[LDGumbelMechanism] = None,
    k_bar: Optional[int] = None
) -> Tuple[int, Dict[int, int]]:
    """
    使用 DP 机制进行多数投票
    
    参数:
        tokens: voter 生成的 token IDs，形状为 (n_voters,)
        dim: 词表大小
        dp_engine: DP 引擎，如果为 None 则使用普通多数投票
        k_bar: DP 候选集大小
        
    返回:
        (选中的 token ID, token -> 票数 映射)
    """
    # 统计每个 token 的票数
    uni_tokens, cnts = tokens.unique(return_counts=True)
    token_votes = {token.item(): vote.item() for token, vote in zip(uni_tokens, cnts)}
    
    if dp_engine is None:
        # 普通多数投票
        max_cnt = torch.max(cnts)
        max_idxs = torch.where(cnts == max_cnt)[0]
        if len(max_idxs) == 1:
            return uni_tokens[max_idxs[0]].item(), token_votes
        else:
            # 平票时随机选择
            idx = max_idxs[torch.randint(len(max_idxs), (1,))[0]]
            return uni_tokens[idx].item(), token_votes
    else:
        # DP 投票
        # dp_engine 返回的是“候选序号”，需要映射回真实 token id
        priv_idx = dp_engine.get_top1(cnts, dim, k_bar=k_bar)
        if priv_idx < len(uni_tokens):
            return uni_tokens[priv_idx].item(), token_votes
        else:
            # 需要从不在候选集中的 token 随机选择
            priv_idx = priv_idx - len(uni_tokens)
            x = torch.ones((dim,), device=tokens.device)
            x[uni_tokens] = 0.
            t = torch.nonzero(x, as_tuple=True)[0][priv_idx]
            return t.item(), None


def ensemble_generate(
    llm,
    prompt_list: List[str],
    max_new_tokens: int = 50,
    dp_engine: Optional[LDGumbelMechanism] = None,
    fail_mode: str = 'stop'
) -> Tuple[List[int], int]:
    """
    DP Ensemble 生成
    
    使用多个 voter 独立生成，然后通过 DP 投票选择每个 token。
    
    参数:
        llm: LLM 实例，需要有 generate_and_tokenize 方法
        prompt_list: 提示词列表，每个 voter 一个
        max_new_tokens: 最大生成的 token 数
        dp_engine: DP 引擎，如果为 None 则使用普通投票
        fail_mode: 失败处理模式 'ld_pate', 'rand', 'stop', 'raise'
        
    返回:
        (生成的 token IDs 列表, exit_status: 0=成功, 1=失败)
    """
    generated_tokens = []
    exit_status = 0
    
    for _ in range(max_new_tokens):
        # 获取所有 voter 的 next token
        next_tokens = llm.generate_tokens(prompt_list, temperature=0, max_tokens=1)
        next_tokens = torch.tensor(next_tokens)
        
        # 确定 token 维度 (词表大小)
        # 如果 llm 提供接口获取，否则使用常见的 32000
        try:
            token_dim = llm.get_vocab_size()
        except:
            token_dim = 32000  # 默认值
        
        if dp_engine is None:
            # 普通投票
            next_token = majority_vote(next_tokens, token_dim)[0]
        else:
            # DP 投票
            try:
                next_token, _ = majority_vote(
                    next_tokens, 
                    token_dim, 
                    dp_engine=dp_engine
                )
            except NotFoundLDTop1:
                if fail_mode == 'stop':
                    print("LD failed. Stop.")
                    exit_status = 1
                    break
                elif fail_mode == 'raise':
                    raise
            except DPExpenseOverflow:
                eps, delta = dp_engine.get_dp_expense()
                print(f"## dp_engine dp eps={eps:.4f}, delta={delta:.4f}")
                exit_status = 1
                break
        
        generated_tokens.append(next_token)
        
        # 更新所有 prompt
        prompt_list = [p + [next_token] for p in prompt_list]
        
        # 检查 EOS
        try:
            eos_id = llm.get_eos_token_id()
            if next_token == eos_id:
                break
        except:
            pass
    
    return generated_tokens, exit_status