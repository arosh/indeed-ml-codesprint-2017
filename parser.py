import subprocess
import lxml.etree

def split_tag_and_description(line):
    names = [
        'part-time-job',
        'full-time-job',
        'hourly-wage',
        'salary',
        'associate-needed',
        'bs-degree-needed',
        'ms-or-phd-needed',
        'licence-needed',
        '1-year-experience-needed',
        '2-4-years-experience-needed',
        '5-plus-years-experience-needed',
        'supervising-job',
    ]
    tags = {}
    for name in names:
        tags[name] = False
    while True:
        line = line.lstrip()
        for name in names:
            if line.startswith(name):
                line = line[len(name):]
                tags[name] = True
                break
        else:
            break
    return tags, line

def tokenize(text):
    out = subprocess.run(
            ['/home/is/sho-ii/devel/indeed-ml/document-preprocessor.sh'],
            input=text.encode('UTF-8'),
            stdout=subprocess.PIPE)
    lines = []
    for line in out.stdout.decode('UTF-8').splitlines():
        lines.append(line.split())
    return lines

def lemmatize(text):
    out = subprocess.run(
            ['/home/is/sho-ii/devel/indeed-ml/pos-tagger.sh'],
            input=text.encode('UTF-8'),
            stdout=subprocess.PIPE)
    xml = lxml.etree.fromstring('<pos>' + out.stdout.decode('UTF-8') + '</pos>')
    lemmas = []
    for sent in xml.xpath('/pos/sentence'):
        l = []
        for word in sent.xpath('./word'):
            l.append(word.attrib['lemma'].lower())
        lemmas.append(l)
    return lemmas
