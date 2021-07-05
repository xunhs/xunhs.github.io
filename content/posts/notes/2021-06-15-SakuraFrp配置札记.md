---
title: 2021-06-15-SakuraFrp配置札记
date: 2021-06-15T03:06:51.000Z
categories:
  - 收藏
tags:
  - 札记
  - 内网穿透
  - frp
slug: 2021-06-15-sakurafrp配置札记
lastmod: '2021-07-05T09:13:50.547Z'
---
> [SakuraFrp](https://www.natfrp.com/)是一个非常好用的内网穿透工具，本文记录配置过程，包含ssh和http两类配置流程。

<!--more-->

------------

<!-- content -->

### 实例一：内网穿透openwrt ssh （树莓派3B+）
#### 创建隧道
![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210615111127.png)
{{< notice info >}}
- 穿透节点：普通高防即可
- 隧道类型：TCP
- 本地地址：注意需填写真实ip地址，`127.0.0.1`此处不适用
- 本地端口：22
{{< /notice >}}


#### 下载软件
- `uname -a`查看内核：
  - 返回：Linux OpenWrt 5.4.124 #0 SMP Fri Jun 11 17:57:31 2021 aarch64 GNU/Linux
  - aarch64即arm64
- 前往https://www.natfrp.com/tunnel/download，选择Linux (arm64)，适用于树莓派3B+


#### 树莓派端运行frpc
- 赋权：`chmod 777 frpc_linux_arm64`
- 运行： `./frpc_linux_arm64 -f 3ebb876549ee0ca6:1407145`

#### 客户端连接
- 验证：前往隧道列表查看隧道是否在线
- 运行：`ssh -p 13694 root@cn-zz-bgp-1.natfrp.cloud`![](https://cdn.jsdelivr.net/gh/xunhs-hosts/pic@master/20210617091512.png)



### 实例二：内网穿透jupyter notebook（win10）
此处仅配置内网穿透，假设系统已经配置好jupyter环境(192.168.123.87:10086)
#### 创建隧道
![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210615112923.png)
{{< notice info >}}
- 穿透节点：国外，可建站
- 隧道：HTTP
- 本地端口：10086
- 绑定域名：jupyter.xunhs.cyou
{{< /notice >}}



#### 下载软件
本次环境为win10，下载win10版本即可

#### 运行frpc
- win10环境下该软件一键安装，配置较为简单
- 根据日志提示，添加DNS记录，本例中使用的[namesilo](https://www.namesilo.com/account_domain_manage_dns.php)![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210615113519.png)
- 等待至少15分钟（半天也是有可能的😂）生效

#### 客户端连接
- 浏览器输入 http://jupyter.xunhs.cyou 即可访问
---

<!-- pic -->
![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210615113826.jpg)
