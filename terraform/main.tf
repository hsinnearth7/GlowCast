terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }

  backend "s3" {
    bucket         = "glowcast-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "glowcast-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "glowcast"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# Networking — VPC
# ---------------------------------------------------------------------------

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.5"

  name = "${var.project_name}-${var.environment}-vpc"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment == "dev" ? true : false
  enable_dns_hostnames = true
  enable_dns_support   = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }

  tags = {
    "kubernetes.io/cluster/${var.project_name}-${var.environment}" = "shared"
  }
}

# ---------------------------------------------------------------------------
# EKS Cluster
# ---------------------------------------------------------------------------

module "eks" {
  source = "./modules/eks"

  cluster_name    = "${var.project_name}-${var.environment}"
  cluster_version = var.eks_cluster_version
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  node_instance_types = var.eks_node_instance_types
  node_desired_size   = var.eks_node_desired_size
  node_min_size       = var.eks_node_min_size
  node_max_size       = var.eks_node_max_size
  node_disk_size      = var.eks_node_disk_size

  environment = var.environment
  tags        = var.tags
}

# ---------------------------------------------------------------------------
# RDS — PostgreSQL
# ---------------------------------------------------------------------------

module "rds" {
  source = "./modules/rds"

  identifier          = "${var.project_name}-${var.environment}"
  engine_version      = var.rds_engine_version
  instance_class      = var.rds_instance_class
  allocated_storage   = var.rds_allocated_storage
  database_name       = var.project_name
  master_username     = var.rds_master_username
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.private_subnets
  allowed_cidr_blocks = module.vpc.private_subnets_cidr_blocks

  multi_az            = var.environment == "prod" ? true : false
  backup_retention    = var.environment == "prod" ? 7 : 1
  deletion_protection = var.environment == "prod" ? true : false

  environment = var.environment
  tags        = var.tags
}

# ---------------------------------------------------------------------------
# ElastiCache — Redis
# ---------------------------------------------------------------------------

module "redis" {
  source = "./modules/redis"

  cluster_id         = "${var.project_name}-${var.environment}"
  node_type          = var.redis_node_type
  num_cache_nodes    = var.environment == "prod" ? 2 : 1
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnets
  allowed_cidr_blocks = module.vpc.private_subnets_cidr_blocks

  environment = var.environment
  tags        = var.tags
}

# ---------------------------------------------------------------------------
# S3 — Model artifacts and data
# ---------------------------------------------------------------------------

module "s3" {
  source = "./modules/s3"

  bucket_prefix = "${var.project_name}-${var.environment}"

  enable_versioning  = true
  enable_encryption  = true
  lifecycle_glacier_days = var.environment == "prod" ? 90 : 30

  environment = var.environment
  tags        = var.tags
}
