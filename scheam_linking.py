def schema_linking(question, columns):
    out = [[0] * 4] * len(columns)
    '''
    Q_list = question.split()
    for i in range(5,0,-1):
        for j in range(len(Q_list)-i):
            word =Q_list[j:j+i]
            if word in columns:
                pass
            elif word in
    '''
    for c_id, col in enumerate(columns):
        for word in question:
            if word in col:
                out[c_id][0] += 1


    sc_vector = []



    return sc_vector