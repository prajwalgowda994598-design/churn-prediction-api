#!/usr/bin/env bash
# deploy/deploy_lambda.sh
# ─────────────────────────────────────────────────────────────────────────────
# Deploys the Churn Prediction API to AWS Lambda using a container image.
#
# Prerequisites:
#   • AWS CLI configured (aws configure or IAM role via EC2/GitHub OIDC)
#   • Docker running
#   • Environment variables set (see below) or passed as CLI args
#
# Usage:
#   chmod +x deploy/deploy_lambda.sh
#   AWS_REGION=us-east-1 ECR_REPO=churn-api LAMBDA_FUNCTION=churn-prediction bash deploy/deploy_lambda.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config (override via env vars or edit here) ───────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ECR_REPO:-churn-prediction-api}"
LAMBDA_FUNCTION="${LAMBDA_FUNCTION:-churn-prediction}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
LAMBDA_ROLE_ARN="${LAMBDA_ROLE_ARN:-}"   # Required for first deploy; optional on update

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "==> AWS Account : ${AWS_ACCOUNT_ID}"
echo "==> Region      : ${AWS_REGION}"
echo "==> ECR URI     : ${ECR_URI}:${IMAGE_TAG}"
echo "==> Lambda Fn   : ${LAMBDA_FUNCTION}"
echo ""

# ── Step 1: Authenticate Docker to ECR ───────────────────────────────────────
echo "[1/5] Authenticating Docker to ECR …"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ── Step 2: Create ECR repo if it doesn't exist ───────────────────────────────
echo "[2/5] Ensuring ECR repository exists …"
aws ecr describe-repositories --repository-names "${ECR_REPO}" \
    --region "${AWS_REGION}" > /dev/null 2>&1 \
  || aws ecr create-repository \
       --repository-name "${ECR_REPO}" \
       --region "${AWS_REGION}" \
       --image-scanning-configuration scanOnPush=true \
       --encryption-configuration encryptionType=AES256

# ── Step 3: Build + tag + push Docker image ───────────────────────────────────
echo "[3/5] Building Docker image …"
docker build --platform linux/amd64 -t "${ECR_REPO}:${IMAGE_TAG}" .

echo "[3/5] Tagging and pushing to ECR …"
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:${IMAGE_TAG}"

# ── Step 4: Create or update Lambda function ──────────────────────────────────
echo "[4/5] Deploying Lambda function …"

LAMBDA_EXISTS=$(aws lambda get-function --function-name "${LAMBDA_FUNCTION}" \
    --region "${AWS_REGION}" 2>&1 || true)

if echo "${LAMBDA_EXISTS}" | grep -q "Function not found"; then
  # ── First-time create ────────────────────────────────────────────────────
  if [[ -z "${LAMBDA_ROLE_ARN}" ]]; then
    echo "[error] LAMBDA_ROLE_ARN must be set for first-time Lambda creation."
    echo "        Create an execution role with AWSLambdaBasicExecutionRole policy."
    exit 1
  fi
  echo "       Creating new Lambda function …"
  aws lambda create-function \
    --function-name "${LAMBDA_FUNCTION}" \
    --package-type Image \
    --code "ImageUri=${ECR_URI}:${IMAGE_TAG}" \
    --role "${LAMBDA_ROLE_ARN}" \
    --timeout 60 \
    --memory-size 1024 \
    --region "${AWS_REGION}" \
    --environment "Variables={LOG_LEVEL=info,PYTHONUNBUFFERED=1}"

  # Add a Function URL for easy testing (no API Gateway needed)
  aws lambda create-function-url-config \
    --function-name "${LAMBDA_FUNCTION}" \
    --auth-type NONE \
    --region "${AWS_REGION}" || true

else
  # ── Update existing function ─────────────────────────────────────────────
  echo "       Updating existing Lambda function …"
  aws lambda update-function-code \
    --function-name "${LAMBDA_FUNCTION}" \
    --image-uri "${ECR_URI}:${IMAGE_TAG}" \
    --region "${AWS_REGION}"

  # Wait for update to complete before fetching URL
  aws lambda wait function-updated \
    --function-name "${LAMBDA_FUNCTION}" \
    --region "${AWS_REGION}"
fi

# ── Step 5: Print the function URL ────────────────────────────────────────────
echo "[5/5] Deployment complete."
FUNCTION_URL=$(aws lambda get-function-url-config \
    --function-name "${LAMBDA_FUNCTION}" \
    --region "${AWS_REGION}" \
    --query FunctionUrl \
    --output text 2>/dev/null || echo "N/A (no Function URL configured)")

echo ""
echo "  Lambda Function : ${LAMBDA_FUNCTION}"
echo "  Function URL    : ${FUNCTION_URL}"
echo "  Health check    : ${FUNCTION_URL}health"
echo ""
echo "  Test prediction :"
echo "  curl -X POST ${FUNCTION_URL}predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d @docs/sample_payload.json"
