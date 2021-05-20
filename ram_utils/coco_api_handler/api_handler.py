import csv


def save_ap_csv(path, ap_data):
    with open(path, mode = 'w') as csv_file:
        fieldnames = ['(AP)IoU-0.50:0.95', '(AP)IoU-0.50', '(AP)IoU-0.75', '(AP)IoU-0.50:0.95(small)','(AP)IoU-0.50:0.95(medium)','(AP)IoU-0.50:0.95(large)','(AR)IoU-0.50:0.95(maxDets=1)', '(AR)IoU-0.50:0.95(maxDets=10)', '(AR)IoU-0.50:0.95(maxDets=100)', '(AR)IoU-0.50:0.95(small)','(AR)IoU-0.50:0.95(medium)','(AR)IoU-0.50:0.95(large)',"Best Model Path","Config"]
        csv_writer = csv.writer(csv_file)
        
        csv_writer.writerow(fieldnames)
        if isinstance(ap_data[0],list):
            csv_writer.writerows(ap_data)
        else:
            csv_writer.writerow(ap_data)
    csv_file.close()