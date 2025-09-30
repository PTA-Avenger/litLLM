# AWS Deployment and Training Guide

## Overview

This guide covers deploying and training the Stylistic Poetry LLM Framework on Amazon Web Services (AWS).

## Quick Start on AWS

### 1. EC2 Deployment

#### Launch GPU Instance for Training

```bash
# Launch p3.2xlarge instance
aws ec2 run-instances \
    --image-id ami-0c94855ba95b798c7 \
    --instance-type p3.2xlarge \
    --key-name first \
    --security-group-ids sg-xxxxxxxxx \
    --user-data file://setup-gpu.sh
```

#### Setup Script (setup-gpu.sh)

```bash
#!/bin/bash
yum update -y
yum install -y docker git python3-pip

# Install Docker and NVIDIA runtime
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
systemctl restart docker

# Clone and setup application
cd /home/ec2-user
git clone https://github.com/your-org/stylistic-poetry-llm.git
cd stylistic-poetry-llm

# Install dependencies
pip3 install -r requirements.txt

# Start application
python3 -m src.main
```

### 2. SageMaker Training

#### Training Script for SageMaker

```python
# sagemaker_train.py
import os
import json
from pathlib import Path
from src.stylometric.training_data import TrainingDataProcessor
from src.stylometric.fine_tuning import FineTuningManager

def train_on_sagemaker():
    # Get SageMaker environment variables
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    train_dir = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')

    # Initialize components
    processor = TrainingDataProcessor()
    trainer = FineTuningManager()

    # Load training data
    with open(f"{train_dir}/training_data.json", 'r') as f:
        training_data = json.load(f)

    # Train model
    model = trainer.prepare_model_for_training("gpt2-medium", "custom_poet")
    result = trainer.train_model(model, training_data)

    # Save model
    trainer.save_trained_model(model, Path(model_dir))
    print(f"Training completed. Loss: {result.final_loss}")

if __name__ == "__main__":
    train_on_sagemaker()
```

#### Launch SageMaker Job

```python
# launch_sagemaker.py
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

def launch_training():
    role = sagemaker.get_execution_role()

    estimator = PyTorch(
        entry_point='sagemaker_train.py',
        role=role,
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        framework_version='1.12.0',
        py_version='py38'
    )

    # Start training
    estimator.fit({'train': 's3://your-bucket/training-data/'})

if __name__ == "__main__":
    launch_training()
```

### 3. Lambda Serverless Deployment

#### Lambda Function

```python
# lambda_function.py
import json
from src.stylometric import PoetryLLMSystem

# Initialize once per container
system = None

def lambda_handler(event, context):
    global system

    if system is None:
        system = PoetryLLMSystem()
        system.initialize()

    # Parse request
    body = json.loads(event['body'])
    prompt = body['prompt']
    poet_style = body.get('poet_style', 'emily_dickinson')

    # Generate poetry
    result = system.generate_poetry_end_to_end(prompt, poet_style)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'poem': result['generated_text'],
            'analysis': result['analysis_results']
        })
    }
```

#### Deploy Lambda

```bash
# Create deployment package
zip -r poetry-llm-lambda.zip src/ lambda_function.py

# Create Lambda function
aws lambda create-function \
    --function-name poetry-llm-generator \
    --runtime python3.9 \
    --role arn:aws:iam::ACCOUNT:role/lambda-role \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://poetry-llm-lambda.zip \
    --timeout 300 \
    --memory-size 3008
```

### 4. ECS Container Deployment

#### Task Definition

```json
{
  "family": "poetry-llm-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "poetry-llm",
      "image": "your-account.dkr.ecr.region.amazonaws.com/poetry-llm:latest",
      "portMappings": [{ "containerPort": 5000 }],
      "environment": [{ "name": "POETRY_LLM_LOG_LEVEL", "value": "INFO" }],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/poetry-llm",
          "awslogs-region": "us-east-1"
        }
      }
    }
  ]
}
```

#### Create ECS Service

```bash
# Create cluster
aws ecs create-cluster --cluster-name poetry-llm-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
    --cluster poetry-llm-cluster \
    --service-name poetry-llm-service \
    --task-definition poetry-llm-task:1 \
    --desired-count 2 \
    --launch-type FARGATE
```

## S3 Storage Integration

### Model Storage

```python
# s3_integration.py
import boto3
from pathlib import Path

class S3ModelManager:
    def __init__(self, bucket_name):
        self.bucket = bucket_name
        self.s3 = boto3.client('s3')

    def upload_model(self, local_path: Path, poet_name: str):
        """Upload trained model to S3"""
        s3_key = f"models/{poet_name}/model.tar.gz"

        # Create tar archive
        import tarfile
        with tarfile.open(f"{poet_name}_model.tar.gz", "w:gz") as tar:
            tar.add(local_path, arcname="model")

        # Upload to S3
        self.s3.upload_file(f"{poet_name}_model.tar.gz", self.bucket, s3_key)
        return f"s3://{self.bucket}/{s3_key}"

    def download_model(self, poet_name: str, local_path: Path):
        """Download model from S3"""
        s3_key = f"models/{poet_name}/model.tar.gz"

        # Download and extract
        self.s3.download_file(self.bucket, s3_key, "model.tar.gz")

        import tarfile
        with tarfile.open("model.tar.gz", "r:gz") as tar:
            tar.extractall(local_path.parent)

# Usage
s3_manager = S3ModelManager("your-poetry-llm-bucket")
s3_manager.upload_model(Path("./models/robert_frost"), "robert_frost")
```

## CloudFormation Template

### Basic Infrastructure

```yaml
# infrastructure.yaml
AWSTemplateFormatVersion: "2010-09-09"
Description: "Poetry LLM Infrastructure"

Resources:
  # S3 Bucket for models and data
  PoetryLLMBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "poetry-llm-${AWS::AccountId}"
      VersioningConfiguration:
        Status: Enabled

  # IAM Role for EC2 instances
  EC2Role:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      Policies:
        - PolicyName: S3Access
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action: ["s3:*"]
                Resource: [!Sub "${PoetryLLMBucket}/*", !Ref PoetryLLMBucket]

  # Security Group
  SecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Poetry LLM Security Group
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0

Outputs:
  BucketName:
    Value: !Ref PoetryLLMBucket
    Export:
      Name: PoetryLLMBucket
```

### Deploy Infrastructure

```bash
# Deploy CloudFormation stack
aws cloudformation create-stack \
    --stack-name poetry-llm-infrastructure \
    --template-body file://infrastructure.yaml \
    --capabilities CAPABILITY_IAM

# Get outputs
aws cloudformation describe-stacks \
    --stack-name poetry-llm-infrastructure \
    --query 'Stacks[0].Outputs'
```

## Monitoring and Cost Optimization

### CloudWatch Monitoring

```python
# monitoring.py
import boto3
import time

class CloudWatchMetrics:
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')

    def put_metric(self, metric_name, value, unit='Count'):
        self.cloudwatch.put_metric_data(
            Namespace='PoetryLLM',
            MetricData=[{
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': time.time()
            }]
        )

    def track_generation(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                self.put_metric('GenerationSuccess', 1)
                return result
            except Exception as e:
                self.put_metric('GenerationError', 1)
                raise
            finally:
                duration = time.time() - start_time
                self.put_metric('GenerationDuration', duration, 'Seconds')
        return wrapper

# Usage
metrics = CloudWatchMetrics()

@metrics.track_generation
def generate_poetry(prompt, poet_style):
    # Your generation logic
    pass
```

### Cost Optimization Tips

1. **Use Spot Instances** for training workloads
2. **Auto Scaling** based on demand
3. **S3 Intelligent Tiering** for model storage
4. **Lambda** for infrequent requests
5. **Reserved Instances** for consistent workloads

### Spot Instance Training

```bash
# Launch spot instance for training
aws ec2 request-spot-instances \
    --spot-price "0.50" \
    --instance-count 1 \
    --launch-specification '{
        "ImageId": "ami-0abcdef1234567890",
        "InstanceType": "p3.2xlarge",
        "KeyName": "your-key-pair",
        "SecurityGroupIds": ["sg-xxxxxxxxx"],
        "UserData": "'$(base64 -w 0 training-script.sh)'"
    }'
```

## Security Best Practices

### IAM Policies

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::poetry-llm-bucket/models/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

### VPC Configuration

```yaml
# Use private subnets for compute resources
PrivateSubnet:
  Type: AWS::EC2::Subnet
  Properties:
    VpcId: !Ref VPC
    CidrBlock: 10.0.1.0/24
    MapPublicIpOnLaunch: false

# NAT Gateway for outbound internet access
NATGateway:
  Type: AWS::EC2::NatGateway
  Properties:
    AllocationId: !GetAtt EIPForNAT.AllocationId
    SubnetId: !Ref PublicSubnet
```

## Troubleshooting

### Common Issues

1. **GPU Not Available**: Ensure NVIDIA drivers and Docker runtime are installed
2. **Memory Issues**: Use smaller models or increase instance memory
3. **S3 Access Denied**: Check IAM permissions and bucket policies
4. **Lambda Timeout**: Increase timeout or use smaller models

### Debug Commands

```bash
# Check GPU availability
nvidia-smi

# View Docker logs
docker logs poetry-llm-container

# Check S3 access
aws s3 ls s3://your-poetry-llm-bucket/

# Monitor CloudWatch logs
aws logs tail /aws/lambda/poetry-llm-generator --follow
```

This guide provides a comprehensive overview of deploying the Poetry LLM Framework on AWS with various deployment options and best practices.
