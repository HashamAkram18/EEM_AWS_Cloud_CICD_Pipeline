
## Authors

- [Hasham Akram](https://github.com/HashamAkram18)
- This repository offers a glimpse about training a machine learning model for effective prediction of energy efficiency factors.
- The model predicts heating and cooling loads for residential buildings, enabling optimization, scenario analysis, sensitivity assessment, comparative studies, integration with management systems, and policy support for energy efficiency


## Website Interface Demo

Here are screenshots demonstrating the interface of the Structures Energy Efficiency project:

### Homepage Model Description

![Homepage](https://github.com/HashamAkram18/AWS-CI-CD-Projects/blob/main/documents/Webapp%20Interface/Screenshot%202024-02-07%20134153.png?raw=true)

### Data Exploration and Model Prediction Interface

![Model Prediction Interface](https://github.com/HashamAkram18/AWS-CI-CD-Projects/blob/main/documents/Webapp%20Interface/Screenshot%202024-02-10%20114751.png?raw=true)




## Software and Account Requirement.
 1. [Github Account](https://github.com/)
 2. [AWS Account](https://aws.amazon.com/free/?gclid=CjwKCAiAt5euBhB9EiwAdkXWOz8dh7VcllPaVXEwzmWN_kLE3axKV3KbjF2T8fZMi8ev5W-Jc5AvpBoCBuMQAvD_BwE&trk=c4f45c53-585c-4b31-8fbf-d39fbcdc603a&sc_channel=ps&ef_id=CjwKCAiAt5euBhB9EiwAdkXWOz8dh7VcllPaVXEwzmWN_kLE3axKV3KbjF2T8fZMi8ev5W-Jc5AvpBoCBuMQAvD_BwE:G:s&s_kwcid=AL!4422!3!637354294239!e!!g!!aws!19043613274!143453611386&all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all)
 3. [Visual studio Code IDE](https://code.visualstudio.com/download)
 4. [Docker Desktop](https://www.docker.com/products/docker-desktop/)
 5. [Docker Hub](https://hub.docker.com/) 
 6. [Git CLI](https://git-scm.com/downloads)
# Structures Energy Efficiency project

Utilizing a statistical machine learning framework, this project aims to analyze the impact of eight input variables (e.g., relative compactness, surface area) on heating load (HL) and cooling load (CL) in residential buildings. Employing classical and non-parametric statistical tools, we'll identify the strongest correlations between input variables and output variables. The project will compare traditional linear regression with advanced non-linear, non-parametric methods like random forests to estimate HL and CL accurately.


## Data

The Dataset you can get through this link: [Energy Efficiency Dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency) 

To import it directly in your code:

```
pip install ucimlrepo

```
```
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
energy_efficiency = fetch_ucirepo(id=242) 
  
# data (as pandas dataframes) 
X = energy_efficiency.data.features 
y = energy_efficiency.data.targets 
  
# metadata 
print(energy_efficiency.metadata) 
  
# variable information 
print(energy_efficiency.variables) 

``` 
- Database:
The given dataset in this project is stored in [Cassandra database](https://www.datastax.com)
see the documentation for [Create and Connect](https://docs.datastax.com/en/astra/astra-db-vector/databases/database-overview.html) data in your Local coding environment.


## Installation

### Requirments
- python 3.8+
- Cassandra Driver (To interact with the Cassandra database, you'll need a suitable driver. The DataStax Python Driver for Apache Cassandra is commonly used for Python projects.)
- pandas 
- numpy 
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook
- Flask / Django
- HTML / CSS / JavaScrip
- Docker

    
## Project Structure
```
Structures Energy Efficiency project
|
|
├───Artifacts
│       .gitignore
│       model.pkl
│       preprocesser.pkl
│       raw.csv
│       test.csv
│       train.csv
│
├───Notebook
│   │   __init__.py
│   │
│   └───Data
│           1 . EDA ENERGY EFFICIENCY.ipynb
│           2. MODEL TRAINING_non_EDA.ipynb
│           3. MODEL TRAINING_EDA.ipynb
│           ANN_model.ipynb
│           cassendra-integration.ipynb
|           secure-connect-hasham-akram-db.zip
|           Hasham_Akram_db-token.json
│           ENB2012_data.csv
│           
│
├───src
│   │   __init__.py
│   │
│   └───MLproject
│       │   __init__.py
│       │   exception.py
│       │   logger.py
│       │   utils.py
│       │
│       ├───components
│       │       __init__.py
│       │       data_ingestion.py
│       │       data_transformation.py
│       │       model_monitoring.py
│       │       model_trainer.py
│       │
│       └───pipelines
│               __init__.py
│               prediction_pipeline.py
│               trainig_pipeline.py
|
|____templates
|        |
|        |________index.html
|        |
|        |________home.html
|
|
|
│   
│   Dockerfile
│   main.py
│   README.md
│   requirements.txt  
│   setup.py
│   template.py
│   webapp.py
```
# Project Name: Structures Energy Efficiency

## Deployment

### Overview
This project is deployed on an AWS EC2 instance using a Docker image stored in Amazon ECR. Continuous Integration and Continuous Deployment (CI/CD) pipelines are implemented using GitHub Actions for automated deployment.

   This projects [CI/CD pipeline](https://github.com/HashamAkram18/AWS-CI-CD-Projects/blob/main/.github/workflows/main.yaml)

### Prerequisites
- An AWS account with permissions to create and manage EC2 instances and ECR repositories.
- Docker installed locally.
- AWS CLI configured with appropriate credentials.
- GitHub repository set up with appropriate access and permissions.
### Deployment Steps
Provide step-by-step instructions on how to deploy the project. Include commands, configurations, or any other relevant information necessary for deployment.

Continuous Integration (CI) Setup

1. **Step 1:** Create a GitHub Actions workflow file in your project repository (e.g., .github/workflows/main.ymal) to define CI jobs.
    ```bash
    name: Continuous Integration

    on:
    push:
        branches: [ main ]

    jobs:
    build:
        runs-on: ubuntu-latest
        steps:
        - name: Checkout code
            uses: actions/checkout@v2

        - name: Build Docker image
            run: |
            docker build -t your-image-name .

    ```

2. **Step 2:** Configure AWS credentials and permissions in GitHub Secrets to allow GitHub Actions to push Docker images to Amazon ECR.

Continuous Deployment (CD) Setup


1. **Step 1:** Add a deployment workflow file (e.g., .github/workflows/main.ymal) to your project repository to define CD jobs.

Example main.ymal:

```
name: Continuous Deployment

on:
  workflow_dispatch:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Login to Amazon ECR
        run: aws ecr get-login-password --region your-region | docker login --username AWS --password-stdin your-account-id.dkr.ecr.your-region.amazonaws.com

      - name: Push Docker image to ECR
        run: |
          docker tag your-image-name:latest your-account-id.dkr.ecr.your-region.amazonaws.com/your-repository-name:latest
          docker push your-account-id.dkr.ecr.your-region.amazonaws.com/your-repository-name:latest

```

2. **Step 2:** Set up an EC2 instance on AWS and install Docker to run your Dockerized application.

## Docker Setup In EC2 commands to be Executed

optional
```
 sudo apt-get update -y
```
```
 sudo apt-get upgrade
```

required
```
 curl -fsSL https://get.docker.com -o get-docker.sh
```
```
 sudo sh get-docker.sh
```
```
 sudo usermod -aG docker ubuntu
```
```
 newgrp docker
```
## Configure EC2 as self-hosted runner:

## Setup github secrets:

- AWS_ACCESS_KEY_ID=

- AWS_SECRET_ACCESS_KEY=

- AWS_REGION = 

- AWS_ECR_LOGIN_URI = 

- ECR_REPOSITORY_NAME = 




## Additional Notes
- Make sure to replace placeholders such as your-image-name, your-account-id, your-region, and your-repository-name with actual values.
- Monitor your CI/CD pipelines in GitHub Actions to ensure successful builds and deployments.
- Configure proper security settings and access controls for your AWS resources and GitHub repository.

