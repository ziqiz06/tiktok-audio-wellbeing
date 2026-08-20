# Audio Spectrogram Transformer (trimmed vendor copy)

This directory is a **trimmed copy** of [YuanGongND/ast](https://github.com/YuanGongND/ast) (BSD-3-Clause, see `LICENSE`), keeping only the `ASTModel` class needed to extract embeddings from AudioSet-pretrained weights. The full upstream repo — training pipeline, ESC-50/SpeechCommands recipes, paper, colab demos — is not reproduced here; see the link above for all of that.

`src/models/ast_models.py` has one functional change from upstream: the hardcoded `../../pretrained_models` relative paths were replaced with an absolute path derived from the file's own location (see the NOTICE comment at the top of the file), so it no longer depends on the caller's working directory.

Used by [`embeddingvector.py`](../../embeddingvector.py) in the parent project to extract 768-dim embeddings for downstream modeling.
