from data_preprocessing import save_processed_data
from model import *

def main():
    save_processed_data()
    model = fitANDsave_model()
    get_testAcc(model)

if __name__ == "__main__":
    main()
