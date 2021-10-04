# covid-texts

## Run Clustering

```bash
$ python3 process.py -i {input-file} -o {output-file}

# e.g.
$ python3 process.py -i data/sample_input.json -o data/sample_output.json
```

Note: The input data structure should be like `data/sample_input.json`. And given that the algorithm does not perform well with short texts, we recommended to only keep items having `more than 20 tokens` in the input dataset.  
