FROM python:3
USER root

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i en_US en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

#install library
RUN python -m pip install jupyterlab
RUN python -m pip install requests
RUN python -m pip install numpy
RUN python -m pip install pandas
#RUN python -m pip install pandas-profiling
RUN python -m pip install scipy

RUN python -m pip install matplotlib
RUN python -m pip install seaborn
#RUN python -m pip install plotly

RUN python -m pip install mglearn
RUN python -m pip install scikit-learn
RUN python -m pip install lightgbm

RUN python -m pip install optuna
RUN python -m pip install joblib
RUN python -m pip install IPython
RUN python -m pip install networkx
RUN python -m pip install xlsxwriter
RUN python -m pip install tornado

RUN python -m pip install torch
RUN python -m pip install torchvision

RUN python -m pip install word2vec

RUN python -m pip install datetime

RUN apt-get install -y vim cron tzdata
#RUN python -m pip install smtplib
#RUN python -m pip install MIMEText
#RUN python -m pip install csv

# change to Japan time
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

