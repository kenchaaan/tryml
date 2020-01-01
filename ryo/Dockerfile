FROM centos:7
COPY ["*.py", "wsgi.wsgi", "requirements.txt", "/var/www/html/"]
RUN yum install httpd httpd-devel python3 python3-devel gcc -y; pip3 install -r /var/www/html/requirements.txt; mkdir /var/www/html/static; chown apache:apache /var/www/html/*
COPY templates/sample.html /var/www/html/templates/
COPY docker/py.conf /etc/httpd/conf.d/
COPY docker/httpd.conf /etc/httpd/conf/
CMD ["/usr/sbin/httpd", "-DFOREGROUND"]