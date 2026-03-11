module "glowcast" {
  source = "../../"

  aws_region  = "us-east-1"
  environment = "dev"

  # VPC — smaller for dev
  vpc_cidr             = "10.0.0.0/16"
  private_subnet_cidrs = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnet_cidrs  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  # EKS — minimal for dev
  eks_cluster_version     = "1.29"
  eks_node_instance_types = ["t3.medium"]
  eks_node_desired_size   = 2
  eks_node_min_size       = 1
  eks_node_max_size       = 4
  eks_node_disk_size      = 30

  # RDS — minimal for dev
  rds_engine_version    = "16.2"
  rds_instance_class    = "db.t3.micro"
  rds_allocated_storage = 10

  # Redis — single node for dev
  redis_node_type = "cache.t3.micro"

  tags = {
    Team        = "ml-engineering"
    CostCenter  = "glowcast-dev"
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
