`hpke_rfc9180.json` is the Base-mode X25519/HKDF-SHA256/AES-256-GCM case from the CFRG HPKE test vectors:
https://github.com/cfrg/draft-irtf-cfrg-hpke/blob/master/test-vectors.json

Only the first encryption is retained. These publicly published values are **test keys**, not project credentials. The test checks the derived AEAD key, nonce and exact ciphertext, independently of the project's round-trip behavior.
