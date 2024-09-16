provider "aws" {
  region = var.aws_region
}

resource "aws_instance" "stock_prediction_app" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name      = var.key_name

  tags = {
    Name = "StockPredictionApp"
  }

  user_data = <<-EOF
              #!/bin/bash
              sudo apt-get update
              sudo apt-get install -y docker.io
              sudo systemctl start docker
              sudo systemctl enable docker
              sudo docker pull ${var.docker_image}
              sudo docker run -d -p 80:5000 ${var.docker_image}
              EOF
}
