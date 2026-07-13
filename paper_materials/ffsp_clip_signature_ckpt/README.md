# FFSP Clip Signature on Checkpoint Rollouts

This figure diagnoses the FFSP weighting rule's clipping behavior. It compares FFSP100 checkpoint rollouts at the source size, the same FFSP100 checkpoint at FFSP50, and the FFSP50 weighting checkpoint at FFSP50.

Main figure: `ffsp_clip_signature_ckpt.png`.

The key point is saturation: roughly half of the pairwise pre-final scores exceed the upper clamp, so final weights can look similar even when the rule depends on low-margin, local rank-neighbor regions.
