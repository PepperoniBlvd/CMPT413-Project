Here is a sample model checkpoint folder. Unzip it in this directory: https://drive.google.com/file/d/1MiLzEXFgsMT8MRWwaiI-MeF9mYm9Oo3y/view?usp=sharing

To evaluate a model, run test.py. The following parameters are available:
* -M: Model name, mandatory. For the given folder, use distilgpt2.
* -c: Checkpoint folder name, mandatory. For the given folder, use checkpoint-11446.
* -b: Batch size. Default is 16.
* -f: Input filename. For the given file, use "small.txt".
* -o: Output filename for the class labels.

For an example command syntax:
`python3 test.py -M "distilgpt2" -c "checkpoint-11446" -b 16 -f "small.txt" -o "small.out"`

-----

To run the training program, do `python3 train.py -M "distilgpt2" -p -e 2 -n 0.1 -b 32`.
This does prefix tuning for a distilgpt2 model, with prefix projection, on 10% of the training data. 
It trains for 2 epochs with a batch size of 32. 
Note that Gemma model only works with LoRA.
More parameters are available using `python3 train.py --help`.