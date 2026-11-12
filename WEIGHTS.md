# Publishing and using the 10000 adapter

Only `stage2-final-0225/checkpoints-10000` is selected for this release. Its tensor SHA256 is:

```text
0ed9694868d2552d243140caf242ef38e0d69f6673a3eec36e4ddc53ed459535
```

The adapter is about 180 MiB. GitHub hosts the code and small demonstrations; the intended model host is Hugging Face. The model owner has selected Apache-2.0 for these weights and `huangwenjun1994/Siphon-LongCat-LoRA` as the target repository. Public upload is pending Hub authentication; this repository ID is not yet an available download.

## Prepare the minimal package

The export tool copies only the standard PEFT tensor/config files, a model card, checksums, and release metadata. It does not copy the approximately 25 GiB of optimizer/full training states or the repeated `transformer/` adapter from the old checkpoint folder. It verifies the selected adapter hash and leaves the original checkpoint unchanged.

```bash
python tools/upload_adapter.py \
  --checkpoint /path/to/checkpoints-10000 \
  --stage-dir outputs/hub-adapter --dry-run

python tools/upload_adapter.py \
  --checkpoint /path/to/checkpoints-10000 \
  --stage-dir outputs/hub-adapter
```

The second command prepares a local draft package only; it does not create a remote repository. Use a new empty staging directory for each export. The config's base-model metadata is normalized to public identifiers without changing the LoRA tensors.

## Upload to your account

Authenticate in your own terminal using `hf auth login`. Do not paste a token into chat or put one in the repository. The following command uses the model owner's selected account and Apache-2.0 weight license:

```bash
python tools/upload_adapter.py \
  --checkpoint /path/to/checkpoints-10000 \
  --repo-id huangwenjun1994/Siphon-LongCat-LoRA \
  --license apache-2.0 \
  --stage-dir outputs/hub-adapter-publish \
  --upload
```

For a separate release, use your own authenticated account or an authorized organization. `--upload` is the only mode that contacts the Hub, creates the model repository if necessary, and uploads the allowlisted adapter files. After upload, put the verified model URL in the main README and check inference with that repository ID.
