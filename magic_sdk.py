import requests
from bs4 import BeautifulSoup
import torch

def scrape_google():
    response = requests.get('https://www.google.com')
    if response.status_code == 200:
        return response.text
    else:
        raise Exception('Error while scraping Google. Status code: {}'.format(response.status_code))

def scrape(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception('Error while scraping {}. Status code: {}'.format(url, response.status_code))

def torch_tensor(data_list):
    return torch.tensor(data_list)

def run_inference(model, input_tensor):
    model.eval()
    with torch.no_grad():
        return model(input_tensor)

def torch_softmax(tensor):
    return torch.softmax(tensor, dim=0)

def display_result_dict(result_dict):
    for key, value in result_dict.items():
        print(f'{key}: {value}')

def print_cow_ascii():
    cow = """
     ^__^                             
     (oo)\_______                   
     (__)\       )\/\             
         ||----w |           
         ||     ||   
    """
    print(cow)