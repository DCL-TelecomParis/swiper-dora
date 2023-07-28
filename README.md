# Swiper and Dora: efficient solutions to weighted distributed problems

Solver for weight reduction problems.
See the [paper](Swiper_and_Dora_TR.pdf) for details and formal definitions.

## Running the solver

On Unix systems, clone the repository and run the following commands inside the repository folder:
```
pip install -r requirements.txt
./main.py swiper --help
```

Solver usage examples:
```
./main.py swiper --tw 1/3 --tn 1/2 ./examples/aptos.dat --speed 1 -v
./main.py swiper --tw 0.3 --tn 1/3 ./examples/tezos.dat --speed 3
./main.py dora --tw 1/3 --tn 1/4 ./examples/filecoin.dat --speed 5
./main.py dora --tw 2/3 --tn 5/8 ./examples/algorand.dat --speed 5
```
