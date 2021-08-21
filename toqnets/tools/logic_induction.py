import torch
from jaclearn.logic.propositional.logic_induction import search
from tqdm import tqdm


def get_logic_formula(inputs, outputs, in_names, f):
    """
    Return searched logic formula
    inputs, outputs are torch tensors
    in_names are the names for input
    f[k] is the set of considered variables for the k-th output
    """
    inputs = inputs.view(-1, inputs.size(-1))
    outputs = outputs.view(-1, outputs.size(-1))
    assert inputs.size(-1) == len(in_names)
    assert outputs.size(-1) == len(f)
    formula = []
    progress_bar = tqdm(range(outputs.size(-1)))
    progress_bar.set_description('Computing logic formulas')
    for k in progress_bar:
        fs = list(sorted(list(f[k])))
        if len(fs) == 0:
            vmin = int(outputs.type(torch.int).min().item())
            vmax = int(outputs.type(torch.int).max().item())
            assert vmax == vmin
            formula.append('True' if vmax > .5 else 'False')
        else:
            logic_inputs = inputs[:, torch.LongTensor(fs)].type(torch.uint8).numpy().astype('uint8')
            logic_in_names = [in_names[fid] for fid in fs]
            logic_outputs = outputs[:, k:k + 1].type(torch.uint8).numpy().astype('uint8')
            check_logic_data(logic_inputs, logic_outputs, logic_in_names)
            formula.append(str(search(logic_inputs, logic_outputs, logic_in_names)))
    return formula


def check_logic_data(inputs, outputs, names):
    n = len(names)
    B = inputs.shape[0]
    assert inputs.shape == (B, n)
    assert outputs.shape == (B, 1)
    ans = {}
    for i in range(B):
        in_str = ''
        for k in range(n):
            if int(inputs[i, k]) == 1:
                in_str += '1'
            elif int(inputs[i, k]) == 0:
                in_str += '0'
            else:
                raise ValueError()
        if int(outputs[i, 0]) == 1:
            out_str = '1'
        elif int(outputs[i, 0]) == 0:
            out_str = '0'
        else:
            raise ValueError()
        if in_str in ans:
            if ans[in_str] != out_str:
                print("Different output for same input")
                raise ValueError()
        ans[in_str] = out_str
    print(ans)
