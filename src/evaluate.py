import numpy as np
import os
import re
import subprocess
import tempfile
import torch
from torch.nn.parallel import DistributedDataParallel
import traceback
import urllib


def beam_search(learn, src_data, beam_size, max_length):
    """
    Beam search with given `beam_size` for the best output sequences of
    maximum length `max_length` given the `src_data` input sequences.
   
    This is vectorized, but uses a simple algorithm that always
    calculates `max_length` tokens instead of stopping at the end of the
    longest sentence.

     It also continues to calculate the probability of outputs even after the
    end of sequence (EOS) token is output.

    `src_data` is expected to be a torch tensor of shape
    (batch_size, max input sentence length).
    """
    batch_size = src_data.shape[0]
    model = learn.model
    if isinstance(model, DistributedDataParallel):
        model = model.module
    offsets = torch.tensor(range(0, batch_size * beam_size * beam_size,
                                 beam_size * beam_size),
                           dtype=torch.int64).to(src_data.device)
    offsets = offsets.unsqueeze(1).repeat(1, beam_size)

    # Peform initial evaluation.
    with torch.no_grad():
        logits = model.eval().predict(src_data)
        vocab_size = logits.shape[1]

    # Select `beam_size` most probable first tokens.
    top_k_total_logits, top_k_idx = torch.topk(logits, beam_size, dim=1)
    tgt_data = top_k_idx.unsqueeze(2)
    assert (list(tgt_data.shape) == [batch_size, beam_size, 1])

    with torch.no_grad():
        for j in range(1, max_length):
            # Create a separate batch for each of the `beam_size` possibilities.
            tgt_dataset = [
                x.squeeze(1) for x in tgt_data.split([1] * beam_size, dim=1)
            ]

            # Predict next tokens by estimating their log probabilities.
            logits = torch.cat([
                model.eval().predict(src_data, tgt_batch).unsqueeze(1)
                for tgt_batch in tgt_dataset
            ],
                               dim=1)
            assert (list(logits.shape) == [batch_size, beam_size, vocab_size])

            # Select `beam_size` most probable next tokens for each example.
            values, indices = torch.topk(logits, beam_size, dim=2)

            # Accumulate total log probability of the entire sequences.
            total_logits = (values + top_k_total_logits.unsqueeze(2).repeat(
                1, 1, beam_size)).view(batch_size, beam_size * beam_size)

            # Select `beam_size` seqs out of `beam_size * beam_size` total.
            top_k_total_logits, top_k_indices_of_indices = \
                torch.topk(total_logits, beam_size, dim=1)
            assert (list(top_k_total_logits.shape) == [batch_size, beam_size])

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
                    index=indices_of_indices[i] // beam_size -
                    i * beam_size).unsqueeze(0)
                for i, t in enumerate(tgt_data.split([1] * batch_size))
            ],
                                    dim=0)

            # Concatenate the new tokens to the end of the existing output seqs.
            tgt_data = torch.cat((old_in_data, top_k_indices.unsqueeze(2)), 2)
            assert (list(tgt_data.shape) == [batch_size, beam_size, j + 1])

    return tgt_data[:, 0, :]


def moses_bleu_score(hypotheses, references, lowercase=False):
    """
    Calculates the BLEU score of `hypotheses` with respect to `references`
    using the Moses library scripts.

    The bleu score script is downloaded as needed.

    Adapted from
    https://pytorchnlp.readthedocs.io/en/latest/
            _modules/torchnlp/metrics/bleu.html
    """
    if isinstance(hypotheses, list):
        hypotheses = np.array(hypotheses)
    if isinstance(references, list):
        references = np.array(references)

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    # Get MOSES multi-bleu script
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(project_dir, 'data', 'mosesdecoder')
    bleu_path = os.path.join(src_dir, 'scripts', 'generic', 'multi-bleu.perl')
    moses_url = ("https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
                 "master/scripts/generic/multi-bleu.perl")
    try:
        if not os.path.isfile(bleu_path):
            os.makedirs(os.path.dirname(bleu_path), exist_ok=True)
            urllib.request.urlretrieve(moses_url, filename=bleu_path)
            os.chmod(bleu_path, 0o755)
    except:
        print("Unable to fetch multi-bleu.perl script")
        print(traceback.format_exception_only(sys.last_type, sys.last_value))
        return None

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

    # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd,
                                               stdin=read_pred,
                                               stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
            bleu_score = np.float32(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
            bleu_score = None

    # Close temp files
    hypothesis_file.close()
    reference_file.close()

    return bleu_score
