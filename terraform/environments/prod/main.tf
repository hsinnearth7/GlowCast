module "glowcast" {
  source = "../../"

  aws_region  = "us-east-1"
  environment = "prod"

  # VPC
  vpc_cidr             = "10.0.0.0/16"
  private_subnet_cidrs = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnet_cidrs  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  # EKS — production-grade
  eks_cluster_version     = "1.29"
  eks_node_instance_types = ["m5.large"]
  eks_node_desired_size   = 3
  eks_node_min_size       = 2
  eks_node_max_size       = 10
  eks_node_disk_size      = 50

  # RDS — multi-AZ, larger instance
  rds_engine_version    = "16.2"
  rds_instance_class    = "db.r6g.large"
  rds_allocated_storage = 50

  # Redis — 2-node cluster
  redis_node_type = "cache.r6g.large"

  tags = {
    Team        = "ml-engineering"
    CostCenter  = "glowcast-prod"
    Compliance  = "soc2"
  }
}

output "eks_cluster_endpoint" {
  value = module.glowcast.eks_cluster_endpoint
}

output "rds_endpoint" {
  value = module.glowcast.rds_endpoint
}

output "redis_endpoint" {
  value = module.glowcast.redis_endpoint
}

output "s3_model_bucket" {
  value = module.glowcast.s3_model_bucket
}
