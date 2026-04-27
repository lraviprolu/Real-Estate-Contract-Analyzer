variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for app storage"
  type        = string
}

variable "ecr_repo_name" {
  description = "Name of the ECR repository"
  type        = string
  default     = "real-estate-analyzer"
}

variable "my_ip" {
  description = "Your local machine IP for SSH access"
  type        = string
}

variable "groq_api_key" {
  description = "Groq API key injected into the container"
  type        = string
  sensitive   = true
}