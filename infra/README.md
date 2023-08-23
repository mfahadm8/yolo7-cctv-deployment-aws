# Yolo7 Sagemaker Inference Endpoint 

## Prerequisites

To deploy the https endpoint alongside microservice, you will need to have AWS SAM CLI and python installed. If you haven't installed it yet, follow the instructions below.

### Installing Dependencies

-  Install AWS SAM CLI: To install AWS SAM CLI please refer to the official AWS SAM CLI documentation [here](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)

- Install Python

## Deployment Via Sam

Follow these steps to deploy the entire api endpoint alongside associated resources:
1. Navigate to the project directory:
2. Execute the following commands and pass the desired password as a parameter:
```bash
cd scripts
.\store_api_pass mysupersecurepass
```
Note: You must pass this password in your headers to the api call with header name: `Auth-Pass`

3. Open the `template.yaml` file and update the following Parameter in as per your requirements:
    - `REDIRECT_URL`: Set it to the desired website link. 
4. The default region of deployment is set to be `us-east-1` in samconfig.toml. Please change that as required.
5. Build and deploy the application using the following command:
```bash
sam sync
```
6. Go to Api Gateway console. Click on `Api-->AwsApiGateway-->Api Keys-->Api`
    - Click on show key and note it down. You must pass this key as `x-api-key   header


# Manually Update Api-Pass and x-api-key
In order to Manually update the `Api-Pass` and `x-api-key header` values, Follow the following instructions

## For Api-Pass
- Sign in to your AWS account and open the AWS Management Console.
- Ensure that the region is set to "US East (N. Virginia)" (us-east-1).
- Navigate to the "Systems Manager" service.
- Click on "Parameter Store" in the left-hand menu.
- Use the search bar to find the API_AUTH_PASSWORD parameter.
- Click on the parameter name to open its details.
- Click the "Edit" button on the top right corner.
- Enter the new password securely in the "Value" field.
- Optionally, add a description if needed.
- Click "Save changes" to update the parameter.

## For x-api-key
- Sign in to your AWS account and open the AWS Management Console.
- Navigate to the API Gateway service.
- In the left-hand menu, click on "APIs".
- Click on the desired API Gateway from the list.
- In the API Gateway dashboard, click on the "API Keys" section in the left-hand menu.
- Click on the "Create API Key" button.
- In the "Name" field, enter a unique name for the API key.
- Under "API Key Source", select "Auto Generate" to let AWS automatically generate the API key value.
- Under "Usage Plans", select the appropriate usage plan for the API key.
- Click the "Save" button to create the API key.
- In the API Gateway dashboard, click on the "Resources" section in the left-hand menu.
- In the resources list, select the top-level resource (usually denoted with a forward slash "/").
- Click on the "Actions" button above the resources list and select "Deploy API".
- In the "Deployment stage" section, choose the desired stage (e.g., "prod").
- Click the "Deploy" button to redeploy the API Gateway with the new API key and configuration.