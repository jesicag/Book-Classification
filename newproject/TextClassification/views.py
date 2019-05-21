from django.shortcuts import render
from TextClassification.MainFunction import main

def index (request) :

    return render (request, 'TextClassification/index.html')

def classificationText(request):
    if request.method =='POST':
        text = request.POST['input_text']
        #hasil = ClassificationText(text)
        cal = main.main(text)
        context = {'cal': cal}
        return render(request,'TextClassification/hasilClassification.html',context)
    return render (request,'TextClassification/index.html')

def svm_index (request) :
    return render (request, 'TextClassification/svm_index.html')

def svm(request):
    if request.method =='POST':
        text = request.POST['input_text2']
        #hasil = ClassificationText(text)
        cal2 = main.svmFunction(text)
        context2 = {'cal2': cal2}
        return render(request,'TextClassification/svm_hasil.html',context2)
    return render (request,'TextClassification/svm_index.html')