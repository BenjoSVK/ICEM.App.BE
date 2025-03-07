import argparse



def main(args):
    pass




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--filename", "-f", 
        type=str, help="Filename to be processed"
        )
    p.add_argument(
        "--model", "-m", 
        default="iedl", 
        type=str, help="Model to be used for processing. Default (iedl)"
        )
    
    main(p.parse_args())



