"""Allow ``python -m siphon`` as a shorthand for the inference command."""

from .inference import main

raise SystemExit(main())
