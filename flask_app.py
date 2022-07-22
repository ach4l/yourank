

# A very simple Flask Hello World app for you to get started with...

from flask import Flask, make_response, request, render_template
import io
import csv
import pandas as pd
from processing import score_fun
import numpy as np
# Pre-requisite - Import the writer class from the csv module
from csv import writer





app = Flask(__name__)


@app.route('/cacheon')
def landing():
    return render_template("landing.html")

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/', methods=["GET", "POST"])
def onepage():
    if request.method == 'GET':
        weight_list = ['4','6','4','6','7','8','3','2','5','2','3','3','2','2','1','1','7']
        df_raw = pd.read_csv("nirf_raw.csv")
        df_to_show = df_raw[['College','Original Rank']]
        df_sorted = df_to_show.sort_values('Original Rank')
        df_sorted.columns = ['College','NIRF Rank']
        df_sorted['Your Rank'] = ' '
        df_sorted['Score'] = ' '
        return render_template('combined.html', weight = weight_list,data=df_sorted.to_html(table_id="example"))
    elif request.method == 'POST':
        # Getting all the parameters of the POST request
        if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
            ip_address = request.environ['REMOTE_ADDR']
        else:
            ip_address = request.environ['HTTP_X_FORWARDED_FOR'] # if behind a proxy
        ss = int(request.form["SS"])
        fsr = int(request.form["FSR"])
        fqe = int(request.form["FQE"])
        fru = int(request.form["FRU"])
        pu = int(request.form["PU"])
        qp = int(request.form["QP"])
        ipr = int(request.form["IPR"])
        fppp = int(request.form["FPPP"])
        gph = int(request.form["GPH"])
        gue = int(request.form["GUE"])
        gms = int(request.form["GMS"])
        gphd = int(request.form["GPHD"])
        rd = int(request.form["RD"])
        wd = int(request.form["WD"])
        escs = int(request.form["ESCS"])
        pcs = int(request.form["PCS"])
        pr = int(request.form["PR"])

        # Storing the parameters in a file
        row_to_write = [ip_address,ss,fsr,fqe,fru,pu,qp,ipr,fppp,gph,gue,gms,gphd,rd,wd,escs,pcs,pr]
        with open('weight_register.csv', 'a', newline='') as f_object:
            # Pass the CSV  file object to the writer() function
            writer_object = writer(f_object)
            # Result - a writer object
            # Pass the data in the list as an argument into the writerow() function
            writer_object.writerow(row_to_write)
            # Close the file object
            f_object.close()

        # Calculating the Weighted Sum based on user-defined parameters
        weight_list = [ss,fsr,fqe,fru,pu,qp,ipr,fppp,gph,gue,gms,gphd,rd,wd,escs,pcs,pr]
        weight_list_string = []
        for i in weight_list:
            weight_list_string.append(str(i))
        df_raw = pd.read_csv("nirf_raw.csv")
        df_sorted = score_fun(weight_list,df_raw)
        df_sorted['sorted_rank'] = np.arange(1,len(df_sorted)+1)
        df_to_show = df_sorted[['College','Original Rank', 'sorted_rank','weighted_sum']]
        df_to_show.columns = ['College', 'NIRF Rank', 'Your Rank', 'Score']
        df_to_show.reset_index()
        return render_template('combined.html', weight = weight_list_string,data=df_to_show.to_html(table_id="example"), scroll = "example")

@app.route('/yourank', methods=["GET"])
def yourankdynamic():
    if request.method == 'GET':
        weight_list =  [4.568052492957178, 7.921044065394207, 7.363862284183401, 6.299990003067894, 7.080145554739771, 9.581761362062704, 4.28906838284929, 5.645234686436368, 9.542470585052179, 4.586837368989013, 8.33852618282252, 3.7175833902720705, 4.122514336544452, 4.04507712913755, 2.6595717760344266, 2.152771258080598, 8.085489141376373]
        weight_list_string = []
        for i in weight_list:
            weight_list_string.append(str(i))
        df_raw = pd.read_csv("nirf_raw.csv")
        df_sorted = score_fun(weight_list,df_raw)
        df_sorted['sorted_rank'] = np.arange(1,len(df_sorted)+1)
        df_to_show = df_sorted[['College','Original Rank', 'sorted_rank','weighted_sum']]
        df_to_show.columns = ['College', 'NIRF Rank', 'Your Rank', 'Score']
        df_to_show.reset_index()
        return render_template('combined.html', weight = weight_list_string,data=df_to_show.to_html(table_id="example"), scroll = "example")


# @app.route('/')
# def form():
#     return render_template("main_page.html")

#@app.route("/login/", methods=["GET", "POST"])
#def login():
#    return render_template("login_page.html")

@app.route('/persorank', methods=["GET","POST"])
def show_rank():
    ss = int(request.form["SS"])
    fsr = int(request.form["FSR"])
    fqe = int(request.form["FQE"])
    fru = int(request.form["FRU"])
    pu = int(request.form["PU"])
    qp = int(request.form["QP"])
    ipr = int(request.form["IPR"])
    fppp = int(request.form["FPPP"])
    gph = int(request.form["GPH"])
    gue = int(request.form["GUE"])
    gms = int(request.form["GMS"])
    gphd = int(request.form["GPHD"])
    rd = int(request.form["RD"])
    wd = int(request.form["WD"])
    escs = int(request.form["ESCS"])
    pcs = int(request.form["PCS"])
    pr = int(request.form["PR"])
    weight_list = [ss,fsr,fqe,fru,pu,qp,ipr,fppp,gph,gue,gms,gphd,rd,wd,escs,pcs,pr]
    df_raw = pd.read_csv("nirf_raw.csv")
    df_sorted = score_fun(weight_list,df_raw)
    df_sorted['sorted_rank'] = np.arange(1,len(df_sorted)+1)
    df_to_show = df_sorted[['College','Original Rank', 'sorted_rank','weighted_sum']]
    df_to_show.columns = ['College', 'NIRF Rank', 'PersoRank', 'Score']
    df_to_show.reset_index()
    return render_template('simple.html',  tables=[df_to_show.to_html(classes='data',index=False,
                       bold_rows = True, border =2,
                       col_space = 100,
                       justify = 'center',
                       na_rep =' ')], titles=['College', 'NIRF Rank', 'Your Rank', 'Your Score'])


@app.route('/persoranktest', methods=["GET","POST"])
def show_rank_one():
    ss = int(request.form["SS"])
    fsr = int(request.form["FSR"])
    fqe = int(request.form["FQE"])
    fru = int(request.form["FRU"])
    pu = int(request.form["PU"])
    qp = int(request.form["QP"])
    ipr = int(request.form["IPR"])
    fppp = int(request.form["FPPP"])
    gph = int(request.form["GPH"])
    gue = int(request.form["GUE"])
    gms = int(request.form["GMS"])
    gphd = int(request.form["GPHD"])
    rd = int(request.form["RD"])
    wd = int(request.form["WD"])
    escs = int(request.form["ESCS"])
    pcs = int(request.form["PCS"])
    pr = int(request.form["PR"])
    weight_list = [ss,fsr,fqe,fru,pu,qp,ipr,fppp,gph,gue,gms,gphd,rd,wd,escs,pcs,pr]
    df_raw = pd.read_csv("nirf_raw.csv")
    df_sorted = score_fun(weight_list,df_raw)
    df_sorted['sorted_rank'] = np.arange(1,len(df_sorted)+1)
    df_to_show = df_sorted[['College','Original Rank', 'sorted_rank','weighted_sum']]
    df_to_show.columns = ['College', 'NIRF Rank', 'Perso Rank', 'Score']
    df_to_show.reset_index()
    return render_template('simple2.html', data=df_to_show.to_html(table_id="example"))






@app.route('/transform', methods=["GET","POST"])
def transform_view():



    roll_no = request.form["roll_no"]
    raw_filename = request.form["raw_filename"]
    info = request.form["info"]
    #print(roll_no)

    f1 = request.files['data_file_1']
    f2 = request.files['data_file_2']

    if not f1:
        return "No prediction file for Dataset 1"
    if not f2:
        return "No prediction file for Dataset 1"

    stream1 = io.StringIO(f1.stream.read().decode("UTF8"), newline=None)
    csv_input1 = list(csv.reader(stream1))
    precision1, accuracy1, results_df1 = roadz_precision(csv_input1)
    precision1 = precision1*10
    accuracy1 = accuracy1*10

    stream2 = io.StringIO(f2.stream.read().decode("UTF8"), newline=None)
    csv_input2 = list(csv.reader(stream2))
    precision2, accuracy2, results_df2 = roadz_precision(csv_input2)
    precision2 = precision2*10
    accuracy2 = accuracy2*10

    merged_df = merge_preds_dumb(results_df1, results_df2)
    no_of_points = len(merged_df)


    merged_precision, merged_accuracy = roadz_precision_df(merged_df)
    score = no_of_points * merged_accuracy
    results_df1.to_csv('database_reports/'+roll_no+'_'+ raw_filename + '_dataset1_' + info + '.csv', index=False)
    results_df2.to_csv('database_reports/'+roll_no+'_'+ raw_filename + '_dataset2_' + info + '.csv', index=False)

    with open('results_raw.csv','a') as fd:
        fd.write(roll_no+"," + str(precision1) +","+ str(accuracy1) +","+ str(precision2) +","+ str(accuracy2) +","+ str(merged_precision) +","+ str(merged_accuracy) +","+str(no_of_points) +","+str(score) +","+ raw_filename +","+info + "\n")


    if merged_accuracy > 90:
        if roll_no == 'Nitesh':
            message = "FUCK YEAH NITESH!"
        elif roll_no == 'Mridul':
            message = "FUCK YEAH MRIDUL!"
    else:
        message = 'Great Going'
    # Plotting submission history
    message =  'Great Going'
    df = pd.read_csv('results_raw.csv')
    df_roll = df.loc[df['Name'] == roll_no]
    acc_list = df_roll['Merged_Accuracy'].values.tolist()
    sub_no_list = range(1,len(acc_list)+1)
    #print(sub_no_list)
    return render_template('accuracy_page.html', accuracy = merged_accuracy, message = message, labels = sub_no_list, values = acc_list,title='Your Submission History', max=100)

@app.route('/lb')
def html_table():
    df = pd.read_csv('results_raw.csv')
    df = df.sort_values(df.columns[1], ascending=False).drop_duplicates([df.columns[0]])
    df = df.reset_index()
    del df['index']
    return render_template('simple.html',  tables=[df.to_html(classes='data',
                       bold_rows = True, border =2,
                       col_space = 100,
                       justify = 'center',
                       na_rep =' ')], titles=df.columns.values)


@app.route('/lbnew', methods=["GET","POST"])
def data_table():
    with open('results_raw.csv','r') as f:
        reader = csv.DictReader(f)
        results = []
        for row in reader:
            results.append(dict(row))
    print(results)


    return render_template('datatables.html', results = results)

@app.route('/chart')
def plot_history():
    a = [1,2,3,4]
    lab = ['a','b','c','d']
    return render_template('bar_chart.html',title='Bitcoin Monthly Price in USD', max=5, values = a, labels = lab)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)




