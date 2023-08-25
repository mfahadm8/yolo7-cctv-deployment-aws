#!/bin/bash

# Set the AWS region and profile
AWS_REGION="ca-central-1"

# Set the parameter name
PARAMETER_NAME="API_AUTH_PASSWORD"

# Read the parameter value from the command-line argument
PARAMETER_VALUE="$1"

if [ -z "$PARAMETER_VALUE" ]; then
  echo "Please provide a value for the parameter."
  exit 1
fi

# Check if the parameter already exists
aws ssm describe-parameters \
  --region "$AWS_REGION" \
  --profile "$AWS_PROFILE" \
  --parameter-filters "Key=Name,Values=$PARAMETER_NAME" \
  --output text > /dev/null 2>&1

if [ $? -eq 0 ]; then
  # Parameter already exists, update its value
  aws ssm put-parameter \
    --region "$AWS_REGION" \
    --profile "$AWS_PROFILE" \
    --name "$PARAMETER_NAME" \
    --value "$PARAMETER_VALUE" \
    --type "SecureString" \
    --overwrite
else
  # Parameter doesn't exist, create it
  aws ssm put-parameter \
    --region "$AWS_REGION" \
    --profile "$AWS_PROFILE" \
    --name "$PARAMETER_NAME" \
    --value "$PARAMETER_VALUE" \
    --type "SecureString"
fi
