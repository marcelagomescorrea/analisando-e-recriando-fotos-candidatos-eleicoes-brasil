# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

run_preprocess_images:
	python -c 'from data_source.main import preprocess; preprocess()'

run_train_pca:
	python -c 'from pca_logic.main import fit; fit()'

run_train_autoencoder:
	python -c 'from autoencoder_logic.main import train; train()'

##### Prediction API - - - - - - - - - - - - - - - - - - - - - - - - -

run_api:
	uvicorn fast_api.api:app --reload

##### Docker - - - - - - - - - - - - - - - - - - - - - - - - -

docker_build:
	docker build -t template-image-api .

docker_run:
	docker run -p 8000:8000 --name api template-image-api

docker_clean:
	docker builder prune

##### GCP - - - - - - - - - - - - - - - - - - - - - - - - -

GCP_PROJECT_ID=bootcamp-368320

DOCKER_IMAGE_NAME=analisando-e-recriando-fotos-candidatos-eleicoes-brasil-backend

# https://cloud.google.com/storage/docs/locations#location-mr
GCR_MULTI_REGION=us.gcr.io

# https://cloud.google.com/compute/docs/regions-zones#available
REGION=us-central1

build_gcr_image:
	docker build -t $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME) .

build_gcr_image_m1:
	docker build --platform linux/amd64 -t $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME) .

run_gcr_image:
	docker run -e PORT=8000 -p 8080:8000 --env-file .env $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME)

push_gcr_image:
	docker push $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME)

deploy_gcr_image:
	gcloud run deploy --image $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME) --platform managed --region $(REGION) --memory '2Gi' --env-vars-file .env.yaml

build_push_deploy_gcr_image:
	make build_gcr_image_m1
	make push_gcr_image
	make deploy_gcr_image
