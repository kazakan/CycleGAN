from pathlib import Path

class BaseLogger:
    def __init__(self):
        pass

    def write(self,values : dict):
        raise NotImplementedError()

class SimpleLogger(BaseLogger):
    def __init__(self,column_names,formats = {}):
        super().__init__()
        self.column_names = column_names
        self.formats = formats

    def write(self, values: dict):
        content = ""
        for cname in self.column_names:
            s = f"{cname} : "
            if cname in values:
                # decide format str
                formatstr = ""
                if cname in self.formats:
                    formatstr = self.formats[cname]
                else:
                    _type = type(values[cname])
                    if _type is float :
                        formatstr = ".6f"
                
                # append column data to row
                s += ("{:"+formatstr+"} | ").format(values[cname])
            else :
                s += "? | "

            content += s
        return content

class ConsoleLogger(BaseLogger):
    def __init__(self,column_names):
        super().__init__()
        self.column_names = column_names

    def write(self, values: dict):
        content = ""
        for cname in self.column_names:
            if cname in values:
                content += f"{cname} : {values[cname] : .6f} | "
        print(content)

class CsvLogger(BaseLogger):
    def __init__(self, column_names,csv_path,writemode="w"):
        super().__init__()
        self.column_names = column_names
        self.csv_path = Path(csv_path)
        self.writemode = writemode

        assert not self.csv_path.is_dir()
            
        with open(self.csv_path,self.writemode) as f:
            header = ','.join(column_names)
            f.write(header+"\n")

        
    def write(self, values: dict):
        content = ""
        for cname in self.column_names:
            if cname in values:
                content += f"{values[cname]},"
            else:
                content += ","
        content = content[:-1] + "\n"

        with open(self.csv_path,'a') as f:
            f.write(content)

if __name__ == "__main__":
    cols = ["epochs","train_loss","val_loss","val_metric"]
    
    consoleLogger = ConsoleLogger(cols)
    csvLogger = CsvLogger(cols,"./tmp/history.csv",'w')

    vals={
        "epochs" : 10,
        "train_loss" : 1000.6,
        "val_loss" : 2030301.4,
        "val_metric" : 0.243324
    }
    consoleLogger.write(vals)
    csvLogger.write(vals)

    vals={
        "epochs" : 20,
        "train_loss" : 10,
        "val_loss" : 5,
        "val_metric" : 0.8
    }
    consoleLogger.write(vals)
    csvLogger.write(vals)

    