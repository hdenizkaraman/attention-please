# Attention Mekanizması
Program dahilinde: Girilen kelime ile bağdaştırılabilecek belirli sayıda diğer kelime kullancıya gösterilir. Geliştirilme aşamasında olduğundan eksiklikler gözlenebilir.

## :floppy_disk: Kurulum
```bash
git clone https://github.com/hdenizkaraman/attention-please.git
pip3 install -r requirements.txt
python3 manage.py
```
Loglar canlı bir şekilde terminalden okunabilir. 
Tarayıcınıza [bu adresi](http://localhost:7680 "bu adresi")  girerek arayüze ulaşabilirsiniz.

### Modeli Eğitmek
Eğitim sekmesinden öncellikle modeli oluşturmak gerek. 
Hâlihazırda **cache** klasörü içerisinde eğitilmiş bir model çıktısı bulunuyor.
Bu çıktı yüklenerek zamandan tasarruf edilebilir.
Eğitilmiş model, yine **cache** klasörü içerisine ilgili UI elementleri ile kaydedilebilir.

### Mekanizmanın Uygulanması
Eğitim tamamlandıktan sonra, ikinci sekmeye geçebilirsiniz. Dilediğiniz kelimeyi yazıp ilgili sonuçları elde edebilirsiniz.

## :pill: Eksiklikler
1. Veriseti dünya siyasetini ilgilendiren konular ile sınırlı. Cumhurbaşkanlarına, ülke isimlerine daha net sonuçlar veriyor.

2. Veriseti tam anlamı ile temizlenmemiş durumda. İlgili kelimeler bazen anlamsız parçalardan oluşabiliyor (rakam veya çekim ekleri gibi).

3. UI kısmı Python ile oluşturulduğundan yetersiz kalmakta.

4. Model son derece basit bir yapı barındırıyor.

## :scroll: Yol Haritası ve Genel Yapılar
> **class Dataset**
Veri HuggingFace ortamından çekiliyor. Temizleme ve tokenizasyon ile bu sınıf ilgileniyor.

> **class SelfAttentionModel**
**forward** metotu ile mekanizma öğrenimi sağlanıyor, **inference** metotu ile skorlar elde ediliyor.

> **class WebInterface**
UI elementleri ve backend arasındaki köprü buradan yönetiliyor.

> **class Manager**
Dosyaların orkestra hâlinde çalışmasını sağlıyor.

## :flashlight: Teknolojiler
![Python](https://img.shields.io/badge/Python-000000?style=for-the-badge&logo=python&logoColor=white)
![Pytorch](https://img.shields.io/badge/Pytorch-de3513?style=for-the-badge&logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-ffe6d0?style=for-the-badge&logo=gradio&logoColor=black)
![Huggingface](https://img.shields.io/badge/HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=white)
