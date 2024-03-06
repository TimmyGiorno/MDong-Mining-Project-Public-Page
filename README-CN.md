## 0. 前言

本项目由 Bingchu Zhao 和 Han Lyu 开发，旨在解决矿场情境下，车辆装载矿物的质量检测问题。

这是一个商业项目，因此没有开源协议。请联系我们或 [三一众力股份有限公司](http://www.sanyzl.com/) 获得更多细节。
- 2024 年 1 月 3 日，该项目已获得中华人民共和国国家版权局计算机软件著作权登记证书。
- **著作权人**：Bingchu Zhao & Han Lyu
- **登记号**：2023SR1743500

<img alt="Software Copyright Certificate.png" height="600" src="assets%2FImages%2FSoftware%20Copyright%20Certificate.png" width="400"/>

我们感谢来自 [内蒙古蒙东矿建有限公司](http://mdkj2018.com/) 和 [河北石家庄三一众力股份有限公司](http://www.sanyzl.com/)
（众力工程机械有限公司）的工程人员与我们的诚力合作。

Bingchu Zhao 和 Han Lyu 平等地对该项目做出了主要贡献和维护：<br>

<a href="https://github.com/TimmyGiorno/MDong-Mining-Project-Public-Page/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TimmyGiorno/MDong-Mining-Project-Public-Page"  alt="Contributors' Avatars"/>
</a>
<br><br>

- Bingchu Zhao, School of Computer Science, Wuhan University 
  - **GitHub 页面**: [@TimmyGiorno](https://github.com/TimmyGiorno)
  - **个人网站**: [timmygiorno.github.io](https://timmygiorno.github.io/)
  - **邮箱**: [albernttimmy@outlook.com](mailto:albernttimmy@outlook.com)
- Han Lyu, College of Control Science and Enginerring, Zhejiang University
  - **GitHub 页面**: [@ShimizuYoshiKazu](https://github.com/ShimizuYoshiKazu)
  - **邮箱**: [3567815517@qq.com](mailto:albernttimmy@outlook.com)
- Bingchu Zhao 和 Han Lyu 已创办了世际线人工智能应用软件（杭州）有限公司，旨在解决工业级别的算法和应用问题。欢迎联系我们讨论潜在的合作。


---
## 1. 项目概述

矿场情境下，我们需要检测每次装载矿物的卡车的装载质量，主要应用为检测矿物装满与否。这个问题可以用深度学习的方法来解决，但是它也面临着：
1. 背景、车型、矿物类型、光线、视角的泛化性问题；
2. 标注数据量不足的固有缺陷；
3. 工业场景对于高性能的需求。

有基于此，我们开发了一个快速、稳健、通用的智能检测系统：
- 我们的应用进行了良好的封装，提供了基于 `Django` 服务的 Web 应用；
- 在我们的算法 Pipeline 中：
  - 上游的预处理阶段: 我们自行训练并微调了高性能的语义分割和识别模型，可以轻松应对工业场景复杂的情景；
  - 下游的分类阶段: 一种现有开源的轻量级的算法，基于小规模数据集即可完成训练。
- 我们的应用有行业领先的性能:
  - 在使用 CPU 计算时只需要平均 13 秒就能完成一张图片的质检；
  - 在使用 GeForce RTX 3070 计算时，只需要 0.7 秒就能完成一张图片的质检；
  - 具体性能测试，请查看后面的章节。

该项目基于 `Python` 3.11 及以上的环境，使用的主要框架与软件包有：`Django`, `Pytorch`, `scikit-learn` 等。对于部署端，建议使用 `Gunicorn` + `Nginx`。

该项目为商业闭源项目，已获中华人民共和国国家版权局计算机软件著作权登记证书。

---

## 2. 项目环境配置

本章节用来介绍如何配置运行该项目所需的 `Python` 环境，`Django` 与深度学习相关的模块，以保证该应用能够在本地进行测试运行。

---
### 2.0. 环境需求与性能

该项目是基于 `Django` 的 Web 应用，通过 `POST` 接口来将矿场质检模型部署到随机生成的 URL 上，再通过向该接口传入 `POST` 请求来获得质检结果。为了使该项目流畅运行，
部署该项目的主机需要有以下的基本条件：
- 基础的 `Python` 解释器版本应为 `>=3.11`；
- `CUDA` 版本应为 `>=11.8`； 
- 至少 6GiB 大小的显存。若显存低于该大小，可能会出现无法在显存中一次性加载所有模型参数进行并行计算的问题，这会导致 `Pytorch` 进行分步计算，从而严重降低响应效率；
- 若部署的主机显卡支持 `Flash Attention` 运算，响应时间会有可观幅度的提高；若不支持，则单张图的响应速度一般在 0.6~1.5 秒之间；
- 建议使用 `Linux` 系统运行与部署该项目。

---
### 2.1. Python 环境配置

在本节中，我们以 `Miniconda`, `CUDA 12.1` 和 `Ubuntu 22.04 LTS` 的系统为例进行从零开始的示例配置。请根据个人需求自行调整各种命令参数。

首先，使用命令行创建名为 `venv` 的虚拟环境：

```shell
conda create --name venv python=3.11
```

之后，激活并进入该环境：

```shell
conda activate venv
```

之后，安装各种 `Python` 库，无需使用 `conda install`，只需要使用 `pip3 install` 即可。

由于直接安装 `Pytorch` 会默认安装 CPU 版本，因此，需要先访问 Pytorch 的 [官网](https://pytorch.org/)，根据你的操作系统选择适当的安装指令。在本示例中，指令为：

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

之后，进入项目目录，安装其余的依赖库：

```shell
cd BASE_DIR # 进入项目根目录，自行调整。
pip3 install -r requirements.txt
```

---
### 2.2. 数据库准备工作

该项目默认使用 MySQL 数据库，库中主要维护与操作的表包括：
1. 账户信息 `User`;
2. 可部署模型的参数行 `Integrator_model`;
3. 图像分割器日志 `Segmentator_log`。

关于数据表的设计细节，请参考 `{BASE_DIR}/quality_detector/models.py` 获得更多信息。


我们以一个新数据库为例部署应用。首先，我们创建名为 `test_db` 的数据库，需要使用 utf-8 编码。首先通过 `MySQL Server` 命令行进入数据库：

```shell
# 在 Terminal 中登录 MySQL Server。确保 MySQL Server 在环境变量中。
mysql -u root -p
Enter password: # 输入密码。

Welcome to the MySQL monitor.  Commands end with ; or \g.
Server version: 8.0.33 MySQL Community Server - GPL
```

顺利进入后，创建数据库即可：

```sql
create database test_db default charset=utf8;
```

之后，在 `{BASE_DIR}/MDProject-release/settings.py` 中的数据库字段中修改数据库配置：

```python
DATABASES = {
    'default':
    {
        'ENGINE': 'django.db.backends.mysql',    # 数据库引擎。
        'NAME': 'test_db',  # 数据库名称。
        'HOST': '127.0.0.1',  # 数据库地址，本机 ip 地址 127.0.0.1。
        'PORT': 3306,  # 端口。
        'USER': 'username',  # 数据库用户名。
        'PASSWORD': 'userpassword',  # 数据库密码。
    }
}
```

若使用 `SQLite` 数据库，则修改数据库字段为以下代码：

```python
 DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "database.sqlite3",  
        # 或 'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
```

- 注意，Django 项目默认的 ORM 数据库使用的是 SQLite。若要链接 MySQL 数据库，需要安装 `pymysql` 软件包。在 `MDProject_Release.__init__py` 中，该模块已进行安装：

```python
import pymysql
pymysql.install_as_MySQLdb()
```

最后，我们进入项目根目录，进行数据库迁移工作，即可完成数据库链接与表的创建：

```shell
cd BASE_DIR # 进入项目根目录，请根据实际情况调整。

# Linux, 以 Ubuntu 22.04 LTS 为例
python manage.py migrate
python manage.py makemigrations quality_detector
python manage.py makemigrations
python manage.py migrate quality_detector
python manage.py migrate
```

前往数据库进行查看，若绑定的数据库中出现了 `Segmentator_log` 和 `Integrator_model` 两个表，则说明数据库准备完毕。
- 若修改 `{BASE_DIR}/quality_detector/models.py` 中定义的 Models 的任何字段，都需要进行数据库迁移，以保证数据同步。
- 若创建失败，则可能是数据库配置失败，请分别检查 `{BASE_DIR}/MDProject-release/settings.py` 中的配置和命令行环境中的 `MySQL Server` 配置。


---
### 2.3. 本地部署测试与使用

在项目的根目录下在终端中键入以下指令：

```shell
python ./manage.py runserver
```

若控制台打印出类似于以下的输出，则说明本地测试部署成功：
```shell
Performing system checks...

System check identified no issues (0 silenced).
January 03, 2024 - 04:26:24
Django version 4.2.7, using settings 'MDProject_release.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

Django 的 `runserver` 功能可以将应用部署在本地以方便测试，访问 URL 为 `http://127.0.0.1:8000/` 。

要进入质检应用，请使用浏览器访问
[http://127.0.0.1:8000/quality_detector/login/](http://127.0.0.1:8000/quality_detector/login/)，
若出现以下界面，则本地部署测试成功。

![Runserver GUI.png](assets%2FImages%2FRunserver%20GUI.png)

在终端中可以使用 CTRL+C 来终止本地部署测试。若需要创建管理员账户，请使用以下的一系列指令进行创建。**该系统的许多功能需要依靠管理员账户来运行，请创建管理员账户以进行服务的正常部署**。

```shell
python .\manage.py createsuperuser
Username (leave blank to use '你的系统用户名'): # 管理员用户名。
Email address: # 邮箱，可以为空。
Password: # 密码。
Password (again): # 再次输入密码。
Superuser created successfully.
```

检查 `test_db` 数据库中的 `auth_user` 表是否已有该管理员账户，若有，则说明创建成功。

---
### 2.4. 深度学习模型文件配置

该软件的上游使用了基于 Segment Anything 的语言分割框架和基于 Meituan YOLO-V6 的图像识别框架，结合了自行设计的中间件，从而完成矿物和卡车的语义分割。为进行图像处理配置，需要将权重模型放在特定的位置来生效。

首先，下载 SAM 模型和经过我们进行微调的 YOLO 模型：
- **SAM Models**：[百度网盘](https://pan.baidu.com/s/1jKz-bh1_RZpOPDmofbAuAQ)；
- **Fine-tuned YOLO Models**：[百度网盘](https://pan.baidu.com/s/1AFKwiHHx8_YNHsOqQWPH-g)。

下载后，在根目录的 `static` 文件夹中创建 `segmentation_models` 文件夹，并在其中创建 `SAM_models` 和 `YOLO_models` 两个文件夹：
- 将下载到的 SAM 模型（名称应为 `sam_vit_XXXX.pth`）放在 `SAM_models` 中；
- 将下载到的 YOLO 模型（名称应为 `best_ckpt.pt`）放在 `YOLO_models` 中。

最后，在 `static` 文件夹中创建 `tmp` 文件夹。项目的部分结构应当如下（忽略部分文件夹）：

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

至此，环境配置完毕。使用刚刚创建的管理员账户进行登录，如果顺利，应当进入以下页面：

![Index.png](assets%2FImages%2FIndex.png)

---
## 3. 应用操作指南

本章节用来介绍如何使用启动后的 `Django` 应用。如果你使用 `runserver` 进行本地测试，则访问应用的网址为：[http://127.0.0.1:8000/quality_detector/login/](http://127.0.0.1:8000/quality_detector/login/)。使用 admin 账号登录后即可进行完整的应用使用。

登录成功后，我们可以看到两个部分：`Segmentator` 和 `Integrator`，现有如下定义：
1. **Segmentator**：完整封装的矿车图像分割器实例。每个应用后台只能同时存在一个 Segmentator 实例。
2. **Integrator**：一种完整封装的，提供直接的质检调用接口的功能聚合器。你可以通过用户页面为模型添加质检的拟合数据，可以直接通过 URL 部署聚合器来传入图片，从而获得质检结果。每个应用后台可以存在多个 Integrator 实例。在实际使用中，多个 Integrator 调用的是同一个 Segmentator 实例。

在底层代码中，图像传入后会经过：1. 图像分割；2. 特征提取；3. 图像分类（合格与不合格）三个环节，这也是 Integrator 名称的由来：Integrator 聚合了图像处理的所有功能，提供外部接口。

---
### 3.1. Segmentator 的创建、部署与应用

进入服务页面后，可以看到如下的 GUI：

![User GUI.png](assets%2FImages%2FIndex.png)

首先，我们需要进行 Segmentator（分割器）的创建。点击 `创建和部署Segmentator` 按钮，可以进入如下的部署界面：

![Segmentator Construction.png](assets%2FImages%2FSegmentator%20Construction.png)

其中：
- **YOLO device** 和 **SAM device** 选项分别为目标识别和语义分割算法使用的设备，默认为 CUDA（GPU）运算；
- **Sam model type** 选项决定 SAM 步骤中使用的模型种类，有 base/large/huge 三种，越大的模型精度越高，但是训练时间也越长。一般情况下我们使用 large 模型，可以在保证足够精度的情况下尽量提升训练速度；
- **Img size** 选项决定输入图片的尺度大小；
- **Conf thres** 和 **Iou thres** 选项为 YOLO 所需的参数；
- **Max det** 选项决定最大检测目标数，默认为 1000，限制 YOLO 模型一次能够输出的目标数量上限；
- **Yolo model file** 和 **Sam model file** 选择两步算法中使用的参数文件，**注意 Sam model file 选项中的文件要和 Sam model type 中相对应**。

在实际使用中，若不知道具体的模型参数设置，只需要使用默认给出的数值即可，无需再做调整。

设置好上述的参数之后，点击 `创建和部署`，若出现如下的界面，则表明分割器部署完毕：

![Segmentator Success.png](assets%2FImages%2FSegmentator%20Success.png)

完成上一节中的步骤之后，点击 `分割图片并查看效果` 按钮，可以进入到如下的界面：

![Segmentator Test.png](assets%2FImages%2FSegmentator%20Test.png)

点击 `选择文件`，选择用于测试语义分割算法效果的图片集，选中后点击 `获取图像分割结果`，等待服务器运算完成后返回结果，即为该图片集在分割算法中运行的效果。
- `是否导出 YOLO 模型` 选项选择后会将目标识别后的结果也一并返回。
- **强烈建议不要每次上传超过 100 张图片进行测试**。

**注意**：在本接口与接下来所有涉及到传入图片压缩包的接口中，都必须要传入 `.zip` 压缩包，并且压缩包的根目录应当是所有图片，而不是一个文件夹，文件夹中包含所有图片。

正确的压缩包格式：
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

错误的压缩包格式：
```
Images.7z/tar.gz

|-- images # 一个文件夹。
|     |-- img1.png
|     |-- img2.png
.     .
.     .
.     .
|     |-- img8.png
|     |-- img9.png
```

若要取消已部署的 Segmentator，点击 `软取消已部署的Segmentator` 或 `硬取消已部署的Segmentator`，即可取消当前部署的 Segmentator：
- 软取消为考虑有无 Integrator 实例部署，若有 Integrator 实例则无法取消；
- 硬取消为不考虑有无 Integrator 实例部署，强制取消部署 Segmentator，需要输入密码。

### 3.2. Integrator 的创建、部署与应用

完成了上一节中的 Segmentator 的部署之后，我们便可以进行下一步的 Integrator 的创建、部署与应用。

点击 `新建 Integrator` 按钮，可以进入到如下界面：

![Integrator Construction.png](assets%2FImages%2FIntegrator%20Construction.png)

其中：
- **Integrator name**：整合器的名称。
- **Description**：整合器的描述和备注（可以为空）。
- **Score model type**：评价模型类型，默认为 XGBoost，也是性能最好的评价模型。

选择好以上的参数后，点击 `创建`，若出现下面的界面，则 Integrator 创建成功：

![Integrator Success.png](assets%2FImages%2FIntegrator%20Success.png)

完成上一小节的步骤后，点击 `编辑与部署Integrator`，进入到以下界面：

![Integrator Selection.png](assets%2FImages%2FIntegrator%20Selection.png)

选择需要部署的 Intergrator，点击 `选择Intergrator`，进入到 Intergrator 的编辑界面。

![Integrator Options.png](assets%2FImages%2FIntegrator%20Options.png)


首先，我们需要添加对分类模型进行拟合的数据。点击 `增加拟合数据`，进入到拟合数据的添加界面，如下：

![Train Data Apply.png](assets%2FImages%2FTrain%20Data%20Apply.png)

将合格的图片集和不合格的图片集分别压缩成两个压缩包，点击选择文件按钮将两组图片集分别加入到训练集中后，点击添加拟合数据，看到合格样本数、不合格样本数和总量与图片集内图片数量相同即为部署成功。系统只会自动检测压缩包中后缀名为图片的文件。

`清空拟合数据` 只能清空所有的拟合数据。**请自行妥善保存拟合评分模型的训练数据**。

在 Integrator 的评分模型拟合完毕后，下方的三个按钮分别可以修改 Integrator 的评分模型类型、评分模型超参数和评分模型的各个特征权重，在实际应用的过程中可以进行调节来满足不同的需求：

![Weight Selection.png](assets%2FImages%2FWeight%20Selection.png)

点击测试模型即可对模型的效果进行测试，过程类似于 Segmentator 的 `分割图片并查看效果`，在此不做赘述。

一切就绪后，点击 `通过 URL 部署评分模型` 即可对该模型进行 URL 部署。部署的链接为：

`服务器 IP/quality_detector/handle_image_post/随机生成的 URL/`。


- 若你使用 `runserver` 本地部署服务器，则链接为：`http://127.0.0.1:8000/quality_detector/handle_image_post/随机生成的 URL/`；
- 若你的服务器 IP 地址为 $123.45.67.89$，在 $8081$ 端口监听，则链接为：
- 若你使用 `runserver` 本地部署服务器，则链接为：`http://123.45.67.89:8081/quality_detector/handle_image_post/随机生成的 URL/`。


---
### 3.3. Integrator URL 接口格式

你可以通过传入 REST 请求来获得质检结果，你的请求格式应为：
1. `POST` 格式；
2. `Body` 的 `form-data` 部分，设置键名为 `image`，值为图片文件。

示例 REST 请求如下：

![POST example.png](assets%2FImages%2FPOST%20example.png)

该接口返回一个 JSON，格式为：

```json
{
    "score_result": 质检结果
}
```

其中，`score_result` 可能有三种结果：`"Unqualified"`、`"Unqualified"`、以及 `"No Mine Detected"`，分别对应 `合格`，`不合格`，`以及没有检测到承载矿物的车辆` 三种结果。

---

## 4. 应用部署
### 4.1. 基于阿里云的 Ubuntu + Gunicorn + Nginx 部署

本章节以阿里云的 4核 vCPU 15GiB NVIDIA T4 GPU 服务器为例，使用 Ubuntu 22.04 LTS 作为操作系统，部署该软件到 8081 端口。由于 `Python 3.11` 版本的 `uWSGI` 存在兼容性问题，因此我们选用 `Gunicorn` 作为 WSGI HTTP 服务器。

首先，在 `requirements.txt` 中并没有 `Gunicorn` 库的依赖，因此需要先安装该库：

```shell
conda activate venv
(venv) pip install gunicorn

gunicorn --version # 查看 gunicorn 版本，测试是否正常安装。
```

在使用前，我们需要首先保证网络连接正常，这需要我们做两方面的设置：

首先，在阿里云控制台开放对应端口。进入服务器控制台的 `安全组`→`管理规则` 部分，授权源对象为 `0.0.0.0/0`，并设置开放端口为 `8081`； 

![Aliyun Group Rule.png](assets%2FImages%2FAliyun%20Group%20Rule.png)

其次，我们需要设置 `MDProject_Release.settings` 中的 `DEBUG` 模式为 `False`，并设置 `ALLOWED_HOSTS` 为 `["*"]`，后续可以根据安全需求进行修改出入规则。

完成以上两步后，使用 `Django` 自带的开发服务器，使用项目根目录的脚本进行网络连接测试：

```shell
(venv) python ./manage.py runserver 0.0.0.0:8081 
```

在命令行中运行这个指令时，Django 会启动一个开发服务器，可以通过浏览器访问 `http://公网服务器IP:8081` 来查看 Django 应用，若能正常进入应用，说明网络连接配置正确。这个服务器主要用于开发和测试，因为它不支持生产环境中的高并发和安全性要求。

之后，键入 `CTRL+C` 关闭开发服务器，测试 `Gunicorn`。进入项目根目录，运行以下代码：

```shell
cd your_project_dir # 进入项目根目录
gunicorn --bind 0.0.0.0:8081 MDProject_release.wsgi
```

同样，若能正常进入应用，说明网络连接配置正确。

为了常态化部署应用，我们需要创建一个 `socket` 服务。首先创建 `socket` 单元文件：

```
sudo vim /etc/systemd/system/gunicorn.socket

# 写入：
[Unit]
Description=gunicorn socket
[Socket]
ListenStream=/run/gunicorn.sock
[Install]
WantedBy=sockets.target
```

之后，创建 `service` 文件：

```
sudo vim /etc/systemd/system/gunicorn.service

# 写入：
[Unit]
Description=gunicorn daemon
Requires=gunicorn.socket
After=network.target  # 在网络服务之后启动。
[Service]
User=user_name # 机器的使用者名。
Group=www-data
WorkingDirectory=project_base_dir  # 项目根目录。
ExecStart=gunicorn_path \  # gunicorn 的位置，位于虚拟环境的 bin/gunicorn 位置。  
          --access-logfile - \
          --workers 1 \
          --bind unix:/run/gunicorn.sock \
          MDProject_release.wsgi:application
[Install]
WantedBy=multi-user.target
```

两个文件都创建好后，运行 Gunicorn socket：

```shell
sudo systemctl start gunicorn.socket
sudo systemctl enable gunicorn.socket
```

检查 Gunicorn socket 是否运作成功，成功则会显示绿色的 `active` 字样：

```shell
sudo systemctl status gunicorn.socket
```

检查 `gunicorn.sock` 是否存在 `/run` 文件夹中：

```shell
file /run/gunicorn.sock
```

如果发现没有在 `/run` 这个文件夹中，或是有其他问题，可以执行下方指令查看 Log：

```shell
sudo journalctl -u gunicorn.socket
sudo journalctl -u gunicorn
```

最后，测试 Server 运行情况，正常则会显示绿色的 `active` 字样：

```shell
sudo systemctl status gunicorn
```

将问题排除后，需要再重新 run 一次 `Gunicorn`：

```shell
sudo systemctl daemon-reload
sudo systemctl restart gunicorn
```

若状态正常，则能通过 `http://公网服务器IP:8081` 访问服务。

最后，我们安装 `Nginx`：

```shell
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install nginx

# 检查 nginx 状态。
sudo systemctl status nginx
sudo systemctl restart nginx
```

配置 Nginx 代理传给 `Gunicorn`：

```
sudo vim /etc/nginx/sites-available/MDProject_Release.conf

# 输入以下内容：
server {
    listen 8081;
    server_name 你的服务器 IP;

    location = /favicon.ico {
        access_log off;
        log_not_found off;
    }

    location /static/ {
        root 你的项目根目录/static/;
    }

    location / {
        include proxy_params;
        proxy_pass http://unix:/run/gunicorn.sock;
    }
}
```

链接配置文件来启动该服务：

```shell
sudo ln -s /etc/nginx/sites-available/MDProject_Release.conf /etc/nginx/sites-enabled  # 创建一个符号链接。
sudo nginx -t  # 检查 Nginx 的配置文件是否有语法错误。
sudo systemctl restart nginx

# 开放防火墙。
sudo ufw allow 'Nginx Full'
```

访问网址并使用 `sudo systemctl status` 来检查各服务状态，若正常，则部署完成。

---
## 5. 工作流程与未来工作

### 5.1. 时间线
- 03/27/2023：草稿代码整理。
- 04/07/2023：第一版配准代码完成。
- 04/17/2023：第一版分割代码完成。
- 04/25/2023：第一版特征提取完成。
- 05/14/2023：基本框架完成。 
- 06/24/2023：决策树预测函数更新。
- 06/25/2023：特征提取函数更新。
- 07/04/2023：决策树预测函数优化。
- 07/15/2023：特征提取函数调试、API接口函数定义。
- 08/23/2023：第一轮性能测试通过。
- 09/14/2023：抽象类封装完成。
- 09/22/2023：凹凸程度可视化函数完成。
- 09/27/2023：Django 框架定义。
- 10/10/2023：用户系统完成。
- 10/25/2023：工程框架更新完成。
- 11/15/2023：分割算法完成。
- 11/27/2023：批处理算法更新。
- 12/05/2023：特征提取器封装完成。
- 12/12/2023：评分模型参数调整接口完成。
- 12/24/2023：评分模型拟合数据管理接口完成。
- 12/28/2023：特征提取器代码重构完成。
- 12/30/2023：聚合器类接口完成。
- 01/01/2024：聚合器类部署接口完成。
- 01/03/2024：大规模代码性能优化与重构。
- 01/05/2024：数据库重构。
- 01/19/2024：面向工业场景的代码重构与测试。
- 01/28/2024：第二轮项目测试通过。
- 03/01/2024：基于 `Ubuntu`+`Gunicorn`+`Nginx` 的部署测试完成，代码小规模重构。
- 03/06/2024：用户手册完成。

### 5.2. 性能测试分析

1. 对于 RTX-3070 显卡本地环境：
- SAM Large: 1.1s average response time per image
- SAM Huge: 1.35s average response time per image

2. 对于 RTX-4090 显卡本地环境：
- SAM Large: 0.4s average response time per image
- SAM Huge: 0.6s average response time per image

3. 对于 NVIDIA T4 GPU 服务器（杭州）上的服务接口：
- SAM Large: 1.7s average response time per image
- SAM Huge: 2.1s average response time per image

在实践中发现只需要使用 SAM Large Model 即可达到理想的效果，无需使用 SAM Huge Model。服务器性能测试基于 POSTMAN 的迭代测试功能。

### 5.3. 未来工作与已知问题
- Time Zone 问题。
- 命令行脚本开发。
- 多线程配置。
- 美观前端页面。

#### 5.3.1. uWSGI 问题

Python 3.11 不能与 `uWSGI` 库很好地兼容，[互联网解决方案](https://jdhao.github.io/2020/07/02/uwsgi_install_use_issue/) 使用 conda 社区安装 `uWSGI`，具有难以修复的 BUG。强烈建议该项目使用 `Gunicorn` 作为 WSGI HTTP Server。以下方案无法解决 `uWSGI` 无法部署的问题。

```bash
conda install -c conda-forge uwsgi
conda install icu=58
```
