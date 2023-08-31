# Sagemaker Async Inference on Video files.

There are two parts to this implementation. 

1. Sagemaker Inference Endpoint Creattion
2. Https Endpoint with Authorization and custom input/output

## 1. Sagemaker Inference Endpoint Creattion
This part can be deployed by executing all cells in the sagemaker.ipynb file. The instructions for each cell and why we need this step are mentioned in the notebook file. Also some test script to invoke the sagemaker endpoint directly has been written and included in the notebook itself as well for local testing.

## Https Endpoint with Authorization and custom input/output
After the successful sagemaker endpoint creation. We can go to the `infra` directory using `cd infra` command and deploy the infrastructure using following command:
```bash
sam sync
```

The detailed instructions for the second part have been documented in the README.md file in `infra` folder.


