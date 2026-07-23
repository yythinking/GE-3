# rag/dp_mechanisms.py
"""
Differential Privacy Mechanism Wrapper
Core DP mechanisms ported from DPRAG project
"""

import numpy as np
import torch
from torch.distributions import Gumbel
from typing import List, Optional, Tuple, Dict, Any


class NotFoundLDTop1(Exception):
    """Raised when LDGumbelMechanism fails to find valid top-1"""
    pass


class DPExpenseOverflow(Exception):
    """Raised when DP privacy budget is exceeded"""
    pass


class LDGumbelMechanism:
    """
    Limit Domain Gumbel Mechanism
    
    Paper: Durfee, D., & Rogers, R. M. (2019). Practical differentially private 
    top-k selection with pay-what-you-get composition. NeurIPS.
    
    This is a core mechanism for token-level Differential Privacy voting.
    Achieves privacy protection by adding Gumbel noise in majority vote.
    
    Parameters:
        eps: Privacy budget epsilon
        delta: Privacy failure probability delta
        target_eps: Total epsilon budget upper bound
        target_delta: Total delta budget upper bound
        k_bar: DP candidate set size
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
        
        self.total_k = 0  # Number of successfully selected tokens
        self.total_queries = 0  # Total query count
        
        self.fail_mode = fail_mode  # 'ld_pate', 'rand', 'stop', 'raise'
    
    def get_top1(self, cnts: torch.Tensor, dim: int, k_bar: Optional[int] = None, sens: float = 2.0) -> int:
        """
        Input histogram (cnts), output top-1 index
        
        Parameters:
            cnts: Token vote count tensor
            dim: Vocabulary size
            k_bar: DP candidate set size (default uses self.k_bar)
            sens: L0 sensitivity (default 2.0)
            
        Returns:
            Selected token ID
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
            # Calculate threshold
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
            # k_bar == dim, degenerate to PATE
            if k_bar > real_len_cnts:
                v_perp = torch.max(gumbel.sample((k_bar - real_len_cnts,))).item()
                k_bar = real_len_cnts
                output_rand_if_fail = True
            else:
                v_perp = -np.inf  # Never fail
        
        # Add Gumbel noise and select
        v_cnts = sorted_cnts[:k_bar] + gumbel.sample((k_bar,)).to(sorted_cnts.device)
        v_max_idx = torch.argmax(v_cnts)
        
        if v_cnts[v_max_idx] >= v_perp:
            # Success
            self.total_k += 1
            self.check_dp_budget()
            return sorted_idxs[v_max_idx].item()
        else:
            # Failure
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
        Get current accumulated DP privacy expense
        
        Returns:
            (eps, delta) tuple
        """
        if total_queries is None:
            total_queries = self.total_queries
        
        # Use simple composition calculation
        k = self.total_k
        l = total_queries
        
        # Take minimum of three bounds
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
        Check if privacy budget is exceeded
        
        Parameters:
            raise_error: Whether to raise exception when exceeded
            verbose: Whether to print detailed information
            
        Returns:
            Whether still within budget
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
    Majority voting using DP mechanism
    
    Parameters:
        tokens: Token IDs generated by voters, shape (n_voters,)
        dim: Vocabulary size
        dp_engine: DP engine, if None then use regular majority vote
        k_bar: DP candidate set size
        
    Returns:
        (Selected token ID, token -> vote count mapping)
    """
    # Count votes for each token
    uni_tokens, cnts = tokens.unique(return_counts=True)
    token_votes = {token.item(): vote.item() for token, vote in zip(uni_tokens, cnts)}
    
    if dp_engine is None:
        # Regular majority vote
        max_cnt = torch.max(cnts)
        max_idxs = torch.where(cnts == max_cnt)[0]
        if len(max_idxs) == 1:
            return uni_tokens[max_idxs[0]].item(), token_votes
        else:
            # Random selection on tie
            idx = max_idxs[torch.randint(len(max_idxs), (1,))[0]]
            return uni_tokens[idx].item(), token_votes
    else:
        # DP voting
        # dp_engine returns "candidate index", need to map back to real token id
        priv_idx = dp_engine.get_top1(cnts, dim, k_bar=k_bar)
        if priv_idx < len(uni_tokens):
            return uni_tokens[priv_idx].item(), token_votes
        else:
            # Need to randomly select from tokens not in candidate set
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
    DP Ensemble Generation
    
    Use multiple voters to independently generate, then select each token via DP vote.
    
    Parameters:
        llm: LLM instance, needs generate_and_tokenize method
        prompt_list: List of prompts, one per voter
        max_new_tokens: Maximum tokens to generate
        dp_engine: DP engine, if None then use regular voting
        fail_mode: Failure handling mode 'ld_pate', 'rand', 'stop', 'raise'
        
    Returns:
        (Generated token IDs list, exit_status: 0=success, 1=failure)
    """
    generated_tokens = []
    exit_status = 0
    
    for _ in range(max_new_tokens):
        # Get next token from all voters
        next_tokens = llm.generate_tokens(prompt_list, temperature=0, max_tokens=1)
        next_tokens = torch.tensor(next_tokens)
        
        # Determine token dimension (vocabulary size)
        # If llm provides interface, use it; otherwise use common 32000
        try:
            token_dim = llm.get_vocab_size()
        except:
            token_dim = 32000  # Default value
        
        if dp_engine is None:
            # Regular voting
            next_token = majority_vote(next_tokens, token_dim)[0]
        else:
            # DP voting
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
        
        # Update all prompts
        prompt_list = [p + [next_token] for p in prompt_list]
        
        # Check EOS
        try:
            eos_id = llm.get_eos_token_id()
            if next_token == eos_id:
                break
        except:
            pass
    
    return generated_tokens, exit_status