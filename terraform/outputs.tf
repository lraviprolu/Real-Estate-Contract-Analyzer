output "ec2_public_ip" {
  description = "Public IP of the EC2 instance"
  value       = aws_instance.app_server.public_ip
}

output "ecr_repository_url" {
  description = "ECR repository URL for pushing Docker images"
  value       = aws_ecr_repository.app_repo.repository_url
}

output "s3_bucket_name" {
  description = "S3 bucket name for app storage"
  value       = aws_s3_bucket.app_storage.id
}

output "app_url" {
  description = "URL to access the Streamlit app"
  value       = "http://${aws_instance.app_server.public_ip}:8501"
}