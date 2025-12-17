import torch
from tinyllm.network.layers import MultiheadSelfAttention


def test_padding_ignores_padded_positions():
    # create a simple MHA with identity projections so q=k=v=x and output_proj is identity
    d_model = 2
    mha = MultiheadSelfAttention(d_model=d_model, num_heads=1, causal=False)
    for name in ("q_proj", "k_proj", "v_proj", "output_proj"):
        proj = getattr(mha, name)
        proj.weight.data = torch.eye(d_model)

    # batch_size=1, seq_len=3, last position will be treated as padding
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [10.0, 10.0]]])

    # when sequence_length=2, the attention should ignore the third position
    out_masked = mha(x, sequence_length=torch.tensor([2]))

    # compare to running on the truncated input (seq_len=2) with no masking
    out_trunc = mha(x[:, :2, :])

    assert torch.allclose(out_masked[:, :2, :], out_trunc, atol=1e-6)

    # ensure that without masking (no sequence_length) the outputs differ
    out_full = mha(x)
    assert not torch.allclose(out_masked[:, :2, :], out_full[:, :2, :])
