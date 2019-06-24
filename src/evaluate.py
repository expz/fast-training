import torch


def beam_search(learn, src_data, beam_size, max_length):
    """
    Beam search with given `beam_size` for the best output sequences given
    the `src_data` input sequences.
    
    This is a simple algorithm that always
    calculates `max_length` tokens instead of stopping early. It also continues
    to calculate the probability of outputs even after a end of sequence (EOS)
    token is output.
    """
    batch_size = src_data.shape[0]
    model = learn.model
    offsets = torch.tensor(
        range(0, batch_size * beam_size * beam_size, beam_size * beam_size),
        dtype=torch.int64).to(src_data.device)
    offsets = offsets.unsqueeze(1).repeat(1, beam_size)

    # Peform initial evaluation.
    with torch.no_grad():
        logits = model.eval()._predict(src_data)
        vocab_size = logits.shape[1]

    # Select `beam_size` most probable first tokens.
    top_k_total_logits, top_k_idx = torch.topk(logits, beam_size, dim=1)
    tgt_data = top_k_idx.unsqueeze(2)
    assert(list(tgt_data.shape) == [batch_size, beam_size, 1])

    with torch.no_grad():
        for j in range(1, max_length):
            # Create a separate batch for each of the `beam_size` possibilities.
            tgt_dataset = [
                x.squeeze(1) for x in tgt_data.split([1] * beam_size, dim=1)
            ]

            # Predict next tokens by estimating their log probabilities.
            logits = torch.cat([
                    model.eval()._predict(src_data, tgt_batch).unsqueeze(1)
                    for tgt_batch in tgt_dataset],
                dim=1)
            assert(list(logits.shape) == [batch_size, beam_size, vocab_size])

            # Select `beam_size` most probable next tokens for each example.
            values, indices = torch.topk(logits, beam_size, dim=2)

            # Accumulate total log probability of the entire sequences.
            total_logits = (
                values + top_k_total_logits.unsqueeze(2).repeat(1, 1, beam_size)
            ).view(batch_size, beam_size * beam_size)

            # Select `beam_size` seqs out of `beam_size * beam_size` total.
            top_k_total_logits, top_k_indices_of_indices = \
                torch.topk(total_logits, beam_size, dim=1)
            assert(list(top_k_total_logits.shape) == [batch_size, beam_size])

            # `torch.take()` requires a 1D list of indices indexing into a 2D
            # tensor, so we must add offsets to each row of indices.
            indices_of_indices = top_k_indices_of_indices + offsets
            top_k_indices = torch.take(
                indices.view(batch_size, beam_size * beam_size),
                indices_of_indices)

            # Select the previous iteration's output seqs that correspond to
            # the newly found `beam_size` most probable next tokens.
            old_in_data = torch.cat([
                t.squeeze(0).index_select(
                    dim=0,
                    index=indices_of_indices[i] // beam_size - i * beam_size
                ).unsqueeze(0)
                for i, t in enumerate(tgt_data.split([1] * batch_size))
            ], dim=0)

            # Concatenate the new tokens to the end of the existing output seqs.
            tgt_data = torch.cat((old_in_data, top_k_indices.unsqueeze(2)), 2)
            assert(list(tgt_data.shape) == [batch_size, beam_size, j+1])

    return tgt_data[:, 0, :]
