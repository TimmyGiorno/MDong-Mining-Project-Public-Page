## 0. Preface

This project was developed by Bingchu Zhao and Han Lyu, aiming to solve the problem of quality detection of trucks loading minerals in mining scenarios.

This is a commercial project, so there is no open-source license. Please contact us or [Sany Zhongli Co., Ltd.](http://www.sanyzl.com/) for more details.
- On January 3, 2024, this project has obtained the Computer Software Copyright Registration Certificate from the National Copyright Administration of the People's Republic of China.
- **Copyright Owners**: Bingchu Zhao & Han Lyu
- **Registration Number**: 2023SR1743500

<img alt="Software Copyright Certificate.png" height="600" src="assets%2FImages%2FSoftware%20Copyright%20Certificate.png" width="400"/>

We would like to thank the engineers from [Inner Mongolia Mengdong Mining Construction Co., Ltd.](http://mdkj2018.com/) and [Hebei Shijiazhuang Sany Zhongli Co., Ltd.](http://www.sanyzl.com/) (Zhongli Engineering Machinery Co., Ltd.) for their sincere cooperation with us.

Bingchu Zhao and Han Lyu have made equal major contributions and maintenance to this project:

<a href="https://github.com/TimmyGiorno/MDong-Mining-Project-Public-Page/graphs/contributors">
 <img src="https://contrib.rocks/image?repo=TimmyGiorno/MDong-Mining-Project-Public-Page"  alt="Contributors' Avatars"/>
</a>

- Bingchu Zhao, School of Computer Science, Wuhan University
  - **GitHub Page**: [@TimmyGiorno](https://github.com/TimmyGiorno)
  - **Personal Website**: [timmygiorno.github.io](https://timmygiorno.github.io/)
  - **Email**: [albernttimmy@outlook.com](mailto:albernttimmy@outlook.com)
- Han Lyu, College of Control Science and Engineering, Zhejiang University
  - **GitHub Page**: [@ShimizuYoshiKazu](https://github.com/ShimizuYoshiKazu)
  - **Email**: [3567815517@qq.com](mailto:albernttimmy@outlook.com)
- Bingchu Zhao and Han Lyu have established Worldline Artificial Intelligence Application Software (Hangzhou) Co., Ltd., aiming to solve industrial-level algorithm and application problems. Welcome to contact us to discuss potential cooperation.

---
## 1. Project Overview

In mining scenarios, we need to detect the loading quality of trucks carrying minerals each time, mainly for detecting whether the minerals are fully loaded or not. This problem can be solved by deep learning methods, but it also faces the following challenges:
1. Generalization issues regarding backgrounds, vehicle types, mineral types, lighting conditions, and viewing angles;
2. Inherent deficiency of insufficient annotated data;
3. High performance requirements in industrial scenarios.

Based on this, we have developed a fast, robust, and universal intelligent detection system:
- Our application is well-packaged, providing a web application based on the `Django` service;
- In our algorithm pipeline:
 - The upstream preprocessing stage: We have trained and fine-tuned high-performance semantic segmentation and recognition models, which can easily handle complex scenarios in industrial environments;
 - The downstream classification stage: An existing open-source lightweight algorithm that can be trained on a small dataset.
- Our application has industry-leading performance:
 - When using CPU computation, it only takes an average of 13 seconds to complete quality inspection for one image;
 - When using GeForce RTX 3070 for computation, it only takes 0.7 seconds to complete quality inspection for one image;
 - For specific performance tests, please refer to the following sections.

This project is based on the `Python` 3.11 and above environment, and the main frameworks and software packages used are `Django`, `Pytorch`, `scikit-learn`, etc. For deployment, it is recommended to use `Gunicorn` + `Nginx`.

This project is a commercial closed-source project and has obtained the Computer Software Copyright Registration Certificate from the National Copyright Administration of the People's Republic of China.

---
## 2. Project Environment Setup

This section introduces how to set up the `Python` environment, `Django`, and deep learning-related modules required to run this project, ensuring that the application can be tested locally.

---
### 2.0. Environment Requirements and Performance

This project is a web application based on `Django`, which deploys the mining quality inspection model to a randomly generated URL through a `POST` interface, and then obtains the quality inspection results by sending `POST` requests to that interface. To ensure the smooth operation of this project, the host for deploying the project needs to meet the following basic conditions:
- The base `Python` interpreter version should be `>=3.11`;
- The `CUDA` version should be `>=11.8`;
- At least 6GiB of VRAM. If the VRAM is lower than this size, it may cause issues where all model parameters cannot be loaded into the VRAM for parallel computation at once, which will cause `Pytorch` to perform step-by-step computation, severely reducing the response efficiency;
- If the GPU on the deployment host supports `Flash Attention` computation, the response time will be significantly improved; if not, the response time for a single image will generally be between 0.6 and 1.5 seconds;
- It is recommended to run and deploy this project on a `Linux` system.

---
### 2.1. Python Environment Configuration

In this section, we will use `Miniconda`, `CUDA 12.1`, and the `Ubuntu 22.04 LTS` system as an example for a zero-start configuration. Please adjust various command parameters according to your personal needs.

First, use the command line to create a virtual environment named `venv`:

```shell
conda create --name venv python=3.11
```

Then, activate and enter the environment:

```shell
conda activate venv
```

Next, install various `Python` libraries, no need to use `conda install`, just use `pip3 install`.

Since directly installing `Pytorch` will default to installing the CPU version, you need to first visit the Pytorch [official website](https://pytorch.org/), and choose the appropriate installation instructions according to your operating system. In this example, the instruction is:

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then, navigate to the project directory and install the remaining dependencies:

```Shell
cd BASE_DIR # Enter the project root directory, adjust as needed.
pip3 install -r requirements.txt
```

---
### 2.2. Database Preparation

This project uses the MySQL database by default, and the main tables maintained and operated in the database include:

User information `User`;
Parameter rows for deployable models `Integrator_model`;
Image segmentation log `Segmentator_log`.
For details on the database table design, please refer to `{BASE_DIR}/quality_detector/models.py` for more information.

Using a new database as an example for application deployment, we first create a database named `test_db`, which needs to use utf-8 encoding. First, enter the MySQL Server through the command line:

```Shell
# Log into MySQL Server in the Terminal. Make sure MySQL Server is in the environment variables.
mysql -u root -p
Enter password: # Enter the password.

Welcome to the MySQL monitor.  Commands end with ; or \g.
Server version: 8.0.33 MySQL Community Server - GPL
```

After successfully entering, create the database:

```sql
create database test_db default charset=utf8;
```

Then, modify the database configuration in the database field in `{BASE_DIR}/MDProject-release/settings.py`:

```python
DATABASES = {
    'default':
    {
        'ENGINE': 'django.db.backends.mysql',    # Database engine.
        'NAME': 'test_db',  # Database name.
        'HOST': '127.0.0.1',  # Database address, local IP address 127.0.0.1.
        'PORT': 3306,  # Port.
        'USER': 'username',  # Database username.
        'PASSWORD': 'userpassword',  # Database password.
    }
}
```

If using the `SQLite` database, modify the database field to the following code:

```python
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "database.sqlite3",  
        # or 'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
```
- Note that the default ORM database used by Django is SQLite. To connect to the `MySQL` database, you need to install the `pymysql` package. In `MDProject_Release.__init__.py`, this module has been installed:

```python
import pymysql
pymysql.install_as_MySQLdb()
```

Finally, we enter the project root directory and perform the database migration work, which can complete the database connection and table creation:

```Shell
cd BASE_DIR # Enter the project root directory, adjust according to the actual situation.

# Linux, using Ubuntu 22.04 LTS as an example
python manage.py migrate
python manage.py makemigrations quality_detector
python manage.py makemigrations
python manage.py migrate quality_detector
python manage.py migrate
```

Go to the database and check if the `Segmentator_log` and `Integrator_model` tables appear in the bound database, indicating that the database preparation is complete.
- If you modify any field defined in `{BASE_DIR}/quality_detector/models.py` for the Models, you need to perform a database migration to ensure data synchronization.
- If the creation fails, it may be due to a database configuration failure. Please check the configuration in `{BASE_DIR}/MDProject-release/settings.py` and the `MySQL` Server configuration in the command line environment separately.

---
### 2.3. Local Deployment Testing and Usage

In the terminal of the project root directory, enter the following command:

```python
python ./manage.py runserver
```

If the console prints output similar to the following, it means the local test deployment is successful:

```shell
Performing system checks...

System check identified no issues (0 silenced).
January 03, 2024 - 04:26:24
Django version 4.2.7, using settings 'MDProject_release.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

Django's `runserver` feature can deploy the application locally for testing purposes, with the access URL being `http://127.0.0.1:8000/`.

To enter the quality inspection application, please use a browser to visit
`http://127.0.0.1:8000/quality_detector/login/`.
If you see the following interface, the local deployment test is successful.

![Runserver GUI.png](assets%2FImages%2FRunserver%20GUI.png)

You can use CTRL+C in the terminal to terminate the local deployment test. If you need to create an admin account, please use the following series of commands to create one. **Many functions of this system require an admin account to run, so please create an admin account for normal service deployment**.

```shell
python .\manage.py createsuperuser
Username (leave blank to use 'your system username'): # Admin username.
Email address: # Email, can be left blank.
Password: # Password.
Password (again): # Enter the password again.
Superuser created successfully.
```

Check the `auth_user` table in the `test_db` database to see if the admin account has been created. If it exists, it means the creation was successful.

---
### 2.4. Deep Learning Model File Configuration

This software uses an image segmentation framework based on Segment Anything and an image recognition framework based on Meituan YOLO-V6, combined with a custom middleware, to complete the semantic segmentation of minerals and vehicles. To configure image processing, the weight models need to be placed in specific locations to take effect.

First, download the SAM model and the fine-tuned YOLO model:
- **SAM Models**：[Baidu Netdisk](https://pan.baidu.com/s/1jKz-bh1_RZpOPDmofbAuAQ)；
- **Fine-tuned YOLO Models**：[Baidu Netdisk](https://pan.baidu.com/s/1AFKwiHHx8_YNHsOqQWPH-g).

After downloading, create a `segmentation_models` folder in the `static` folder in the root directory, and create two folders inside: `SAM_models` and `YOLO_models`:
- Place the downloaded SAM model (named `sam_vit_XXXX.pth`) in SAM_models;
- Place the downloaded YOLO model (named `best_ckpt.pt`) in YOLO_models.

Finally, create a `tmp` folder in the `static` folder. The structure of the project should be as follows (ignoring some folders):

```
BASE_DIR/
|-- MDProject_release/
|-- quality_detector/
|-- static/
|     |-- segmentation_models/
|     |      |-- SAM_models/
|     |      |      |-- sam_vit_XXX.pth
|     |      |-- YOLO_models/
|     |      |      |-- best_ckpt.pt
|     |-- styles/
|     |-- tmp/
|-- templates/
|-- manage.py
|-- requirements.txt
```

After that, the environment configuration is complete. Use the admin account you just created to log in, and if successful, you should enter the following page:

![Index.png](assets%2FImages%2FIndex.png)

---
## 3. Application Operation Guide

This section is used to introduce how to use the launched `Django` application. If you use `runserver` for local testing, the application URL is: `http://127.0.0.1:8000/quality_detector/login/`. Log in with the admin account to access the complete application usage.

After successful login, we can see two parts: `Segmentator` and `Integrator`, with the following definitions:
1. **Segmentator**: A fully encapsulated instance of the mine vehicle image segmenter. Each application backend can only have one Segmentator instance at the same time.
2. **Integrator**: A fully encapsulated functional aggregator that provides a direct quality inspection call interface. You can add fitting data for the model through the user interface, and you can directly deploy the aggregator through the URL to input images and obtain quality inspection results. Each application backend can have multiple Integrator instances. In actual usage, multiple Integrators call the same Segmentator instance.

At the underlying code level, after the image is input, it will go through: 1. Image segmentation; 2. Feature extraction; 3. Image classification (qualified or unqualified). This is also the origin of the name Integrator: Integrator aggregates all image processing functions and provides external interfaces.

---
### 3.1. Creation, Deployment, and Application of Segmentator

After entering the service page, you can see the following GUI:

![User GUI.png](assets%2FImages%2FIndex.png)

First, we need to create a Segmentator. Click the `Create and Deploy Segmentator` button to enter the following deployment interface:

![Segmentator Construction.png](assets%2FImages%2FSegmentator%20Construction.png)

Where:

- The **YOLO device** and **SAM device** options respectively specify the devices used for object detection and semantic segmentation algorithms, with the default being CUDA (GPU) computation;
- The **Sam model type** option determines the model type used in the SAM step, with three options: base/large/huge. The larger the model, the higher the accuracy, but the longer the training time. Generally, we use the large model, which can maintain sufficient accuracy while maximizing training speed as much as possible;
- The **Img size** option determines the input image scale;
- The **Conf thres** and **Iou thres** options are parameters required by YOLO;
- The **Max det** option determines the maximum number of detection targets, defaulting to 1000, limiting the upper bound of the number of targets that the YOLO model can output at once;
- The **Yolo model file** and **Sam model file** options select the parameter files used by the two-step algorithms. **Note that the file in the Sam model file option must correspond to the Sam model type option**.

In actual usage, if you are unsure about the specific model parameter settings, you can simply use the default values provided, without further adjustment.

After setting the above parameters, click `Create and Deploy`. If you see the following interface, it means the segmenter has been deployed successfully:

![Segmentator Success.png](assets%2FImages%2FSegmentator%20Success.png)

After completing the steps in the previous section, click the `Segment Images and View Results` button to enter the following interface:

![Segmentator Test.png](assets%2FImages%2FSegmentator%20Test.png)

Click `Choose File`, select the image set to test the effect of the semantic segmentation algorithm, click `Get Image Segmentation Results` after selection, and wait for the server to complete the computation and return the results, which will be the effect of running the image set on the segmentation algorithm.
- Selecting the `Export YOLO Model` option will also return the object detection results.
- **It is strongly recommended not to upload more than 100 images for testing at a time**.

**Note**: In this interface and all subsequent interfaces involving uploading image archives, you must upload `.zip` archives, and the root directory of the archive should contain all images, not a folder with all images inside.

Correct archive format:

```
Images.zip
|-- img0.png
|-- img1.png
|-- img2.png
.
.
.
|-- img8.png
|-- img9.png
```

Incorrect archive format:

```
Images.7z/tar.gz

|-- images # A folder!
|     |-- img1.png
|     |-- img2.png
.     .
.     .
.     .
|     |-- img8.png
|     |-- img9.png
```

To cancel the deployed Segmentator, click `Soft Cancel Deployed Segmentator` or `Hard Cancel Deployed Segmentator`:
- Soft cancel considers whether there are any Integrator instances deployed, and if there are, it cannot be canceled;
- Hard cancel does not consider whether there are any Integrator instances deployed, and forcibly cancels the deployment of Segmentator, requiring a password input.

### 3.2. Creation, Deployment, and Application of Integrator

After completing the deployment of Segmentator in the previous section, we can proceed to the next step of creating, deploying, and applying the Integrator.

Click the `New Integrator` button to enter the following interface:

![Integrator Construction.png](assets%2FImages%2FIntegrator%20Construction.png)

Where:
- **Integrator name**: The name of the integrator.
- **Description**: The description and notes for the integrator (can be left blank).
- **Score model type**: The evaluation model type, defaulting to XGBoost, which is also the best-performing evaluation model.

After selecting the above parameters, click `Create`. If you see the following interface, the Integrator has been created successfully:

![Integrator Success.png](assets%2FImages%2FIntegrator%20Success.png)

After completing the steps in the previous section, click `Edit and Deploy Integrator` to enter the following interface:

![Integrator Selection.png](assets%2FImages%2FIntegrator%20Selection.png)

Select the Integrator you want to deploy, click `Select Integrator`, and enter the Integrator editing interface.

![Integrator Options.png](assets%2FImages%2FIntegrator%20Options.png)

First, we need to add data for fitting the classification model. Click `Add Fitting Data` to enter the fitting data addition interface, as shown below:

![Train Data Apply.png](assets%2FImages%2FTrain%20Data%20Apply.png)

Compress the qualified image set and the unqualified image set into two separate archives, click the `Choose File` button to add the two image sets to the training set, and click `Add Fitting Data`. If the number of qualified samples, unqualified samples, and total quantity matches the number of images in the image sets, it means the deployment was successful. The system will automatically detect files with image extensions in the archives.

`Clear Fitting Data` can only clear all fitting data. **Please backup the training data for fitting the evaluation model yourself**.

After the Integrator's scoring model has been fitted, the three buttons below can be used to modify the Integrator's scoring model type, scoring model hyperparameters, and the weights of each feature of the scoring model, which can be adjusted during practical application to meet different requirements:

![Weight Selection.png](assets%2FImages%2FWeight%20Selection.png)

Click `Test Model` to test the effect of the model, the process is similar to `Segment Images and View Results` of Segmentator, and will not be elaborated here.

When everything is ready, click `Deploy Scoring Model via URL` to deploy the model via a URL. The deployment link is:

`Server IP/quality_detector/handle_image_post/randomly generated URL/`.

- If you use `runserver` to deploy the server locally, the link will be: `http://127.0.0.1:8000/quality_detector/handle_image_post/randomly generated URL/`;
- If your server IP address is $123.45.67.89$, and it listens on port $8081$, the link will be: `http://123.45.67.89:8081/quality_detector/handle_image_post/randomly generated URL/`.

---
### 3.3. Integrator URL Interface Format

You can obtain the quality inspection results by sending REST requests. Your request format should be:
1. `POST` format;
2. In the `form-data` part of the `Body`, set the key name to `image` and the value to the image file.

Example REST request:

![POST example.png](assets%2FImages%2FPOST%20example.png)

This interface returns a JSON in the following format:

```json
{
    "score_result": Quality inspection result
}
```

Where `score_result` can have three possible results: `"Unqualified"`, `"Unqualified"`, and `"No Mine Detected"`, corresponding to Qualified, Unqualified, and No vehicle carrying minerals detected, respectively.

---

## 4. Application Deployment Guide

### 4.1. Ubuntu + Gunicorn + Nginx Deployment Based on Alibaba Cloud

This section uses an Alibaba Cloud server with 4 vCPU, 15GiB NVIDIA T4 GPU, and Ubuntu 22.04 LTS as an example, deploying this software to port 8081. Since there are compatibility issues with `uWSGI` in the `Python 3.11` version, we choose `Gunicorn` as the WSGI HTTP server.

First, there is no dependency on the Gunicorn library in `requirements.txt`, so we need to install this library first:

```shell
conda activate venv
(venv) pip install gunicorn

gunicorn --version # Check the gunicorn version, test if it is installed correctly.
```

Before using it, we need to ensure that the network connection is working properly, which requires us to make two settings:

First, open the corresponding port in the Alibaba Cloud console. Go to the server console's `Security Group` -> `Manage Rules` section, authorize the source object as `0.0.0.0/0`, and set the open port to `8081`;

![Aliyun Group Rule.png](assets%2FImages%2FAliyun%20Group%20Rule.png)

Next, we need to set the `DEBUG` mode in `MDProject_Release.settings` to `False`, and set `ALLOWED_HOSTS` to `["*"]`, which can be modified later according to security requirements.

After completing the above two steps, use Django's built-in development server and the script in the project root directory to test the network connection:

```shell
python manage.py runserver 0.0.0.0:8081
```

When running this command in the command line, Django will start a development server, and you can access `http://public_server_IP:8081` in a browser to view the Django application. If you can enter the application normally, it means the network connection configuration is correct. This server is mainly used for development and testing, as it does not support high concurrency and security requirements in production environments.

Then, press `CTRL+C` to close the development server and test `Gunicorn`. Go to the project root directory and run the following code:

```shell
cd your_project_dir # Enter the project root directory
gunicorn --bind 0.0.0.0:8081 MDProject_release.wsgi
```

Similarly, if you can enter the application normally, it means the network connection configuration is correct.

To deploy the application in a persistent manner, we need to create a `socket` service. First, create the `socket` unit file:

```
sudo vim /etc/systemd/system/gunicorn.socket

# Enter:
[Unit]
Description=gunicorn socket
[Socket]
ListenStream=/run/gunicorn.sock
[Install]
WantedBy=sockets.target
```

Then, create the `service` file:

```
sudo vim /etc/systemd/system/gunicorn.service

# Enter:
[Unit]
Description=gunicorn daemon
Requires=gunicorn.socket
After=network.target  # Start after the network service.
[Service]
User=user_name # The machine's user name.
Group=www-data
WorkingDirectory=project_base_dir  # Project root directory.
ExecStart=gunicorn_path \  # Location of gunicorn, in the bin/gunicorn location of the virtual environment.
          --access-logfile - \
          --workers 1 \
          --bind unix:/run/gunicorn.sock \
          MDProject_release.wsgi:application
[Install]
WantedBy=multi-user.target
```

After creating both files, run the Gunicorn socket:

```Shell
sudo systemctl start gunicorn.socket
sudo systemctl enable gunicorn.socket
```

Check if the Gunicorn socket is working successfully, if successful, it will display the green `active` text:

```Shell
sudo systemctl status gunicorn.socket
```

Check if `gunicorn.sock` exists in the `/run` folder:

```Shell
file /run/gunicorn.sock
```

If you find that it is not in the `/run` folder, or if there are other problems, you can execute the following command to view the log:

```Shell
sudo journalctl -u gunicorn.socket
sudo journalctl -u gunicorn
```

After resolving the issues, you need to run `Gunicorn` again:

```Shell
sudo systemctl daemon-reload
sudo systemctl restart gunicorn
```

If the status is normal, you can access the service via `http://public_server_IP:8081`.

Finally, we install `Nginx`:

```Shell
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install nginx

# Check nginx status.
sudo systemctl status nginx
sudo systemctl restart nginx
```

Configure Nginx to proxy to `Gunicorn`:

```
sudo vim /etc/nginx/sites-available/MDProject_Release.conf

# Enter the following content:
server {
    listen 8081;
    server_name your server IP;

    location = /favicon.ico {
        access_log off;
        log_not_found off;
    }

    location /static/ {
        root your project root directory/static/;
    }

    location / {
        include proxy_params;
        proxy_pass http://unix:/run/gunicorn.sock;
    }
}
```

Link the configuration file to start the service:

```Shell
sudo ln -s /etc/nginx/sites-available/MDProject_Release.conf /etc/nginx/sites-enabled  # Create a symbolic link.
sudo nginx -t  # Check if the Nginx configuration file has any syntax errors.
sudo systemctl restart nginx

# Open the firewall.
sudo ufw allow 'Nginx Full'
```

Access the URL and use sudo `systemctl status` to check the status of each service. If normal, the deployment is complete.

---
## 5. Workflow and Future Work

### 5.1. Timeline

- **03/27/2023**: Draft code organization.
- **04/07/2023**: First version of alignment code completed.
- **04/17/2023**: First version of segmentation code completed.
- **04/25/2023**: First version of feature extraction completed.
- **05/14/2023**: Basic framework completed.
- **06/24/2023**: Decision tree prediction function updated.
- **06/25/2023**: Feature extraction function updated.
- **07/04/2023**: Decision tree prediction function optimized.
- **07/15/2023**: Feature extraction function debugging, API interface function definition.
- **08/23/2023**: First round of performance testing passed.
- **09/14/2023**: Abstract class encapsulation completed.
- **09/22/2023**: Concavity/convexity visualization function completed.
- **09/27/2023**: Django framework definition.
- **10/10/2023**: User system completed.
- **10/25/2023**: Engineering framework updated.
- **11/15/2023**: Segmentation algorithm completed.
- **11/27/2023**: Batch processing algorithm updated.
- **12/05/2023**: Feature extractor encapsulation completed.
- **12/12/2023**: Scoring model parameter adjustment interface completed.
- **12/24/2023**: Scoring model fitting data management interface completed.
- **12/28/2023**: Feature extractor code refactored.
- **12/30/2023**: Integrator class interface completed.
- **01/01/2024**: Integrator class deployment interface completed.
- **01/03/2024**: Large-scale code performance optimization and refactoring.
- **01/05/2024**: Database restructuring.
- **01/19/2024**: Code refactoring and testing for industrial scenarios.
- **01/28/2024**: Second round of project testing passed.
- **03/01/2024**: Ubuntu + Gunicorn + Nginx deployment testing completed, minor code refactoring.
- **03/06/2024**: User manual completed.

---
### 5.2. Performance Testing Analysis

1. For the local environment with RTX-3070 GPU:
- SAM Large: 1.1s average response time per image
- SAM Huge: 1.35s average response time per image
2. For the local environment with RTX-4090 GPU:
- SAM Large: 0.4s average response time per image
- SAM Huge: 0.6s average response time per image
3. For the service interface on the NVIDIA T4 GPU server (Hangzhou):
- SAM Large: 1.7s average response time per image
- SAM Huge: 2.1s average response time per image

In practice, we found that using the SAM Large Model is sufficient to achieve the desired effect, and there is no need to use the SAM Huge Model. The server performance testing is based on the iterative testing function of POSTMAN.

---
### 5.3. Future Work and Known Issues

- Time Zone issue.
- Command line script development.
- Multi-threading configuration.
- Visually appealing front-end pages.

#### 5.3.1. uWSGI Compatibility Issue

Python 3.11 cannot be well compatible with the `uWSGI` library, and the internet solution of using the conda community to install `uWSGI` has difficult-to-fix bugs. It is strongly recommended that this project uses `Gunicorn` as the WSGI HTTP Server. The following solution cannot solve the problem of `uWSGI` failing to deploy.

```Shell
conda install -c conda-forge uwsgi
conda install icu=58
```