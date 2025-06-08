# coding=UTF-8

from pypdf import PdfReader
import optparse

def print_meta(file_name):
    pdf = PdfReader(file_name)
    print(f'[*] PDF MetaData For: {file_name}')
    for key, value in pdf.metadata.items():
        print(f'[+] {key}: {value}')

def main():
    parser = optparse.OptionParser('usage: %prog -F <PDF file name>')
    parser.add_option('-F', dest='file_name', help='specify PDF file name')
    options, _ = parser.parse_args()

    if not options.file_name:
        parser.print_help()
    else:
        print_meta(options.file_name)

if __name__ == '__main__':
    main()