import argparse

import torch
import torch.nn.functional as F

from rl4co.models.rl.reinforce.preference_losses import pl_loss, po_loss


def ptp_po_loss(reward: torch.Tensor, log_likelihood: torch.Tensor, alpha: float):
    preference = (reward[:, :, None] > reward[:, None, :]).float()
    logp_pair = alpha * (log_likelihood[:, :, None] - log_likelihood[:, None, :])
    pf_log = torch.log(torch.sigmoid(logp_pair))
    return -(pf_log * preference).mean()


def ptp_pl_loss(reward: torch.Tensor, log_likelihood: torch.Tensor, alpha: float):
    sorted_idx = reward.sort(dim=1, descending=True).indices
    logp = alpha * log_likelihood
    max_logp = logp.max(1, keepdim=True).values
    logp = logp - max_logp
    exp_logp = torch.exp(logp)
    one_hot = F.one_hot(sorted_idx, num_classes=reward.size(1)).float()
    till_mat = torch.tril(torch.ones_like(one_hot))
    sum_exp = (till_mat @ one_hot @ exp_logp.unsqueeze(-1)).squeeze(-1)
    return torch.mean(torch.log(exp_logp) - torch.log(sum_exp))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--pomo", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    reward = torch.randn(args.batch, args.pomo, device=device)
    log_likelihood = torch.randn(args.batch, args.pomo, device=device)

    ptp_po = ptp_po_loss(reward, log_likelihood, args.alpha)
    rl4co_po, _ = po_loss(reward, log_likelihood, alpha=args.alpha)

    ptp_pl = ptp_pl_loss(reward, log_likelihood, args.alpha)
    rl4co_pl_ptp = pl_loss(reward, log_likelihood, alpha=args.alpha, impl="ptp")
    rl4co_pl_stable = pl_loss(reward, log_likelihood, alpha=args.alpha, impl="stable")

    print("po_loss diff:", (ptp_po - rl4co_po).abs().max().item())
    print("pl_loss diff (ptp impl):", (ptp_pl - rl4co_pl_ptp).abs().max().item())
    print(
        "pl_loss diff (stable impl):", (ptp_pl - rl4co_pl_stable).abs().max().item()
    )


if __name__ == "__main__":
    main()
