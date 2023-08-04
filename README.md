# Swiper and Dora: efficient solutions to weighted distributed problems

Solver for weight reduction problems.
See the [paper](Swiper_and_Dora_TR.pdf) for details and formal definitions.

## Running the solver

On Unix systems, clone the repository and run the following commands inside the repository folder:
```bash
pip install -r requirements.txt
./main.py swiper --help
./main.py dora --help
```

Solver usage examples (see the help message for more details):
```bash
# -v enables verbose logging. -vv enables debug logging.
./main.py swiper --tw 1/3 --tn 1/2 ./examples/aptos.dat --speed 1 -v
# --tw and --tn can be fractions or decimals
./main.py swiper --tw 0.3 --tn 1/3 ./examples/tezos.dat --speed 3
./main.py dora --tw 1/3 --tn 1/4 ./examples/filecoin.dat --speed 5
./main.py dora --tw 2/3 --tn 5/8 ./examples/algorand.dat --speed 5
# --gas-limit and --soft-memory-limit help ensure timely results, even with --speed 1.
# The solver's output is deterministic and platform-independent, even with these parameters.
./main.py dora --tw 2/3 --tn 5/8 ./examples/algorand.dat --speed 1 --gas-limit 10B --soft-memory-limit 4G
```
