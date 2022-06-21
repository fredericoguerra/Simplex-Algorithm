from lib2to3.pgen2.literals import simple_escapes
import numpy as np
from sympy.solvers.inequalities import reduce_rational_inequalities
from sympy import Matrix, symbols

np.seterr(divide='ignore', invalid='ignore')

class PrimalOperators:
    def __init__(self):
        self.has_two_phases = False
        self.z = 0

    def set_pivot_column(a : np.array, has_two_phases :False, number_artificial_vars: int):
        if has_two_phases and a[-1,:-1].min() >=0:
            min_col_element_index = np.argmin(a[-2,:-(number_artificial_vars+1)])
        else:
            min_col_element_index = np.argmin(a[-1,:-1])
        col_pivot = a[:, min_col_element_index]
        return min_col_element_index, col_pivot
    
    def set_pivot_row(self, a : np.array, pivot_column: np.array, has_two_phases: False):
        self.has_two_phases = has_two_phases
        if has_two_phases:
            b = a.shape[0] - 2
        else: 
            b = a.shape[0] - 1
        last_column = a[0:b,-1]
        last_column_divided = np.divide(last_column, pivot_column[0:b])
        #print(last_column_divided, min_row_element_index)
        min_index = np.argmin(last_column_divided[last_column_divided>=0])
        #if self.past_pivot_index >= 0 and min_index == self.past_pivot_index:
        #    last_column_divided[min_index] = 999999
        #    min_index = np.argmin(last_column_divided)
        min_row_element_index = min_index
        row_pivot = a[min_row_element_index,:]
        #print(self.past_pivot_index) 
        self.past_pivot_index = min_row_element_index
        return min_row_element_index, row_pivot

    def is_negative(a: np.array, has_two_phases: False, number_artificial_vars: int):
        if np.amin(a[-1,:]) < 0:
            return True
        else:
            if has_two_phases:
                if np.amin(a[-2,:-(number_artificial_vars+1)]) < 0:
                    return True
                else:
                    return False
            else:
                return False

    def pivot_operations(self, a: np.array, has_two_phases: False, number_artificial_vars : int):
        col_pivot_index, col_pivot = PrimalOperators.set_pivot_column(a, has_two_phases=has_two_phases, number_artificial_vars = number_artificial_vars)
        row_pivot_index, row_pivot= PrimalOperators.set_pivot_row(self, a= a, pivot_column=col_pivot, has_two_phases=has_two_phases)
        
        ele_pivot = a[row_pivot_index, col_pivot_index]
        print('------------------------------------------------')
        print(f'Pivot row: {a[row_pivot_index,:]}\nPivot Column: {a[:,col_pivot_index]}\nPivot Element: {ele_pivot}')
        print('------------------------------------------------')
        new_row_pivot = row_pivot/ele_pivot
        a[row_pivot_index,:] = new_row_pivot

        for row in range(0,a.shape[0]):
            if row != row_pivot_index:
                a[row,:] = (new_row_pivot * (-a[row,col_pivot_index]))+a[row,:]
        return a

    def get_results(self, a: np.array, obj: str):
        if self.has_two_phases:
            self.z = a[-2,-1]
        else:
            self.z = a[-1,-1]
        b = a[:-1,-1]
        results = {}
        if obj == 'min':
            results['Z'] = -self.z
        else:
            results['Z'] = self.z
        for index in range(0,a.shape[0]-1):
            for j in range(0,a.shape[0]-1):
                if a[index][j] == 1:
                    results[f'X_{index+1}'] = b[index]
        
        print('------------------------------------------------')
        print('                  RESULTS')
        print('------------------------------------------------')
        for var, res in results.items():
            print(var, '=', res)
        print('------------------------------------------------')

class DualOperators:
    def __init__(self):
        pass

    def get_pivot_row(self, a: np.array) -> int:
        return np.argmin(a[:-1,-1])

    def get_pivot_column(self, a: np.array, row_index: int) -> int:
        aux_row = -(a[-1,:])/a[row_index,:]
        return np.ma.masked_invalid(aux_row[:self.n_add_vars]-1).argmin()
    
    def pivot_table(self, a: np.array, row_index: int, col_index: int) -> np.array:
        ele_pivot = a[row_index, col_index]
        a[row_index,:] = a[row_index,:]/(ele_pivot)
        for row in range(0, a.shape[0]):
            if row != row_index:
                a[row,:] = (-(a[row,col_index])*a[row_index,:])+a[row,:]
        return a

    def is_negative(self, a: np.array):
        if np.amin(a[:-1,-1]) < 0:
            return True
        else:
            return False

    def get_results(self, a: np.array):
        z = a[-1,-1]
        b = a[:-1,-1]
        results = {}
        results['Z'] = -z
        for r_index in range(0,a.shape[0]-1):
            x_index = np.where(a[r_index, 0:self.n_dec_var] == 1)[0][0] + 1
            results[f'X_{x_index}'] = b[r_index]
        
        print('------------------------------------------------')
        print('                  RESULTS')
        print('------------------------------------------------')
        for var, res in results.items():
            print(var, '=', res)
        print('------------------------------------------------')

class Simplex:
    def __init__(self) -> None:
        self.obj_primal = ''
        self.fun_primal = ''
        self.obj_dual = ''
        self.fun_dual = ''
        self.cons_primal = ''
        self.cons_dual = ''
        self.n_add_vars = 0
        self.vars_primal = ''
        self.signals_primal = []
        self.signals_dual = []
        self.var_constraints_list = []
        self.cons_primal_list = []
        self.artificial_vars = []
        self.c_artificial_vars = []
        self.has_two_phases = False
        self.identity_matrix = []
        self.W_primal = []
        self.needs_artificial = []
        self.past_pivot_index = -99999
        self.var_basic_index = []
        self.var_nonbasic_index = []
        self.table_primal_initial = []
        self.B_ = []

        self.A_primal = []
        self.b_primal = []
        self.c_primal = []

        self.A_dual = []
        self.b_dual = []
        self.c_dual = []

    def show_primal(self):
        print('_________________________________')
        print('PRIMAL')
        print('--------')
        print(f'{self.obj_primal} Z = {self.fun_primal}')
        print(f'\ns.a:\n{self.cons_primal}\n\nand\n\n{self.vars_primal}')
        print('--------')
        print(f'Decision variables coeficients array (c):\n{self.c_primal}')
        print(f'Coeficients Matrix (A):\n{self.A_primal}')
        print(f'Restrictions Matrix (b):\n{self.b_primal}')
        print('_________________________________')

    def show_dual(self):
        print('_________________________________')
        print('DUAL')
        print('-------')
        print(f'{self.obj_dual} Z = {self.fun_dual}')
        print(f'\ns.a:\n{self.cons_dual}')
        print('--------')
        print(f'Decision variables coeficients array (c):\n{self.c_dual}')
        print(f'Coeficients Matrix (A):\n{self.A_dual}')
        print(f'Restrictions Matrix (b):\n{self.b_dual}')
        print('_________________________________')
    
    def parse_fun_primal(self):
        vars = self.fun_primal.replace(' ','').split('+')
        for var in vars:
            self.c_primal.append(float(var.split('*')[0]))
        self.b_dual = np.array(self.c_primal).T
        self.c_primal += [0] * self.n_add_vars
        self.c_primal = np.array(self.c_primal)
        if self.obj_primal == 'min':
            self.c_primal*=-1

    def parse_constraints(self, cons: list):
        self.cons_primal_list = cons
        self.signals_primal = []
        for index in range(0,len(cons)):
            if cons[index].find('<=') > -1:
                self.signals_primal.append('<=')
                inequality = '<='
                if self.obj_primal == 'max':
                    self.cons_dual+=f'y_{index+1} >= 0\n'
                elif self.obj_primal == 'min':
                    self.cons_dual+=f'y_{index+1} <= 0\n'
            elif cons[index].find('>=') > -1:
                self.signals_primal.append('>=')
                inequality = '>='
                if self.obj_primal == 'max':
                    self.cons_dual+=f'y_{index+1} <= 0\n'
                elif self.obj_primal == 'min':
                    self.cons_dual+=f'y_{index+1} >= 0\n'
            elif cons[index].find('=') > -1:
                self.signals_primal.append('=')
                inequality = '='
                self.cons_dual+=f'y_{index+1}, free\n'
            else:
                print('error: could not find equation signal. please use <=, >= or =')
                break
            coefs = cons[index].split(inequality)[0].split('+')
            b = cons[index].split(inequality)[1]
            self.b_primal.append(float(b))
            for coef in coefs:
                self.A_primal.append(float(coef.split('*')[0]))
        cols = int(len(self.A_primal)/len(self.b_primal))
        rows = len(self.b_primal)
        
        self.A_primal = np.array(self.A_primal).reshape((rows,cols))
        self.A_primal_cp = self.A_primal
        self.identity_matrix = np.identity(len(self.b_primal))
        for i in range(0, len(self.signals_primal)):
            if self.signals_primal[i] == '>=':
                self.has_two_phases = True
                self.A_primal[i]*=1
                self.b_primal[i]*=1
        
    def get_dual_function(self):
        self.n_dec_var = len(self.c_dual) - 1
        self.c_dual = self.b_primal + [0]*len(self.b_dual)
        for index in range(0,len(self.b_primal)):
            self.fun_dual += str(self.b_primal[index]) + f'*y_{index+1} + '
        self.fun_dual = self.fun_dual[:-3]
        self.A_dual = self.A_primal_cp.T

        self.cons_dual+='\nand\n\n'
        for i in range(0, self.A_dual.shape[0]):
            for j in range(0, self.A_dual.shape[1]):
                self.cons_dual+=(f'{self.A_dual[i][j]}*y_{j+1} + ')
            pos= self.cons_dual.rfind('+')
            if pos > -1:
                if self.var_constraints_list[i].find('>=') > -1:
                    inequality_signal = '>='
                elif self.var_constraints_list[i].find('<=') > -1:
                    inequality_signal = '<='
                elif self.var_constraints_list[i].find('free') > -1:
                    inequality_signal = '='
                else:
                    print('error: missing constraint signal')
                    break
                self.cons_dual = self.cons_dual[:pos] + inequality_signal + self.cons_dual[pos + 1:]
            self.cons_dual+=(f'{self.c_primal[i]}')
            self.cons_dual+=('\n')

    def primal(self):
        self.c = self.parse_fun_primal()
        self.show_primal()
    
    def dual(self):
        self.get_dual_function()
        self.show_dual()

    def objective_function(self, obj: str, fun: str):
        self.obj_primal = obj
        self.fun_primal = fun
        if self.obj_primal == 'max':
            self.obj_dual = 'min'
        elif self.obj_primal == 'min':
            self.obj_dual = 'max'

    def add_constraints(self, constraints: list):
        self.n_add_vars = len(constraints)
        for cons in constraints:
            self.cons_primal+=(f'\n{cons}')        
        self.parse_constraints(cons = constraints)
        for signal in self.signals_primal:
            if signal == '>=':
                self.needs_artificial.append(1)
                self.artificial_vars.append(-1)
                self.c_artificial_vars.append(0)
            elif signal == '<=':
                self.artificial_vars.append(0)
                self.needs_artificial.append(0)
            elif signal == '=':
                self.needs_artificial.append(1)
                self.artificial_vars.append(0)
        
        self.artificial_vars = np.array([self.artificial_vars])
        if sum(self.artificial_vars[0]) < -1:
            self.artificial_vars = -1*np.identity(abs(sum(self.artificial_vars[0])))

        if self.has_two_phases:
            self.A_primal = np.concatenate((self.A_primal, self.artificial_vars.transpose()),axis=1)
            A_zeros = np.zeros((self.A_primal.shape[0],self.A_primal.shape[1]))
            self.A_primal = np.concatenate((self.A_primal, self.identity_matrix.transpose()),axis=1)
            self.W_primal = np.zeros([len(self.A_primal[0])])
            A_zeros = np.concatenate((A_zeros,-self.identity_matrix.transpose()),axis=1)
            for i in range(0, self.A_primal.shape[0]):
                if self.needs_artificial[i] > 0:
                    self.W_primal += self.A_primal[i]
                    self.W_primal += A_zeros[i]
            self.W_primal = -np.array([self.W_primal])
        else:
            self.A_primal = np.concatenate((self.A_primal, self.identity_matrix),axis=1)
    
    def add_var_constraints(self, var_constraints: list):
        self.var_constraints_list = var_constraints
        for var_con in var_constraints:
            self.vars_primal+=(f"{var_con}\n")

    def run_primal(self):
        self.c_primal = self.c_primal.reshape(1,len(self.c_primal))
        self.c_primal *= -1
        
        if self.has_two_phases:
            self.c_artificial_vars = np.array([self.c_artificial_vars])
            self.c_primal = np.concatenate((self.c_primal, self.c_artificial_vars), axis=1)
            print(self.A_primal,self.c_primal)
            self.A_primal = np.concatenate((self.A_primal, self.c_primal),axis=0)
            self.A_primal = np.concatenate((self.A_primal, self.W_primal),axis=0)
            last_b_list = [a * b for a,b in zip(self.b_primal, self.needs_artificial)]
            last_b_ele = -sum(last_b_list)
            self.table_primal = self.A_primal
        else:
            self.table_primal = np.concatenate((self.A_primal, self.c_primal),axis=0)
        self.b_primal = np.array(self.b_primal)
        diff = self.table_primal.shape[0]-self.b_primal.shape[0]
        b_add = np.array([0]*diff)
        self.b_primal = np.concatenate((self.b_primal,b_add),axis=0).astype(np.float)
        self.b_primal = self.b_primal.reshape(len(self.b_primal),1)

        self.table_primal = np.concatenate((self.table_primal,self.b_primal),axis=1)
        if self.has_two_phases:
            self.table_primal[-1,-1] = last_b_ele
        self.table_primal_initial = self.table_primal
        print('\n\n------------------------------------------------')
        print('PRIMAL TABLE:\n', self.table_primal)
        self.table_primal = PrimalOperators.pivot_operations(self, a = self.table_primal, has_two_phases=self.has_two_phases, number_artificial_vars = sum(self.needs_artificial))
        index = 1
        print('------------------------------------------------')
        print(f'{index}º ITERATION')
        self.table_primal = np.round(self.table_primal, 2)
        print('\n',self.table_primal)
        print('------------------------------------------------')
        index_primal = 1
        while PrimalOperators.is_negative(a = self.table_primal, has_two_phases=self.has_two_phases, number_artificial_vars = sum(self.needs_artificial)):
            self.table_primal = PrimalOperators.pivot_operations(self, a = self.table_primal,has_two_phases=self.has_two_phases, number_artificial_vars = sum(self.needs_artificial))
            index += 1
            print(f'{index}º ITERATION')
            self.table_primal = np.round(self.table_primal, 2)
            print('\n',self.table_primal)
            print('------------------------------------------------')
            index_primal += 1
            if index_primal == 10:
                break

        PrimalOperators.get_results(self, a=self.table_primal, obj = self.obj_primal)

    def run_dual(self):
        self.c_dual = np.array(self.c_dual).reshape(1,len(self.c_dual))
        self.A_dual = np.concatenate((-1*self.A_dual,np.identity(self.A_dual.shape[0])),axis=1)
        self.table_dual = np.concatenate((self.A_dual,self.c_dual),axis=0)
        self.b_dual = -1*np.array(self.b_dual)
        diff = self.table_dual.shape[0]-self.b_dual.shape[0]
        b_add = np.array([0]*diff)
        self.b_dual = np.concatenate((self.b_dual,b_add),axis=0).astype(np.float)
        self.b_dual = self.b_dual.reshape(self.b_dual.shape[0],1)
        self.table_dual = np.concatenate((self.table_dual,self.b_dual),axis=1)

        self.table_dual = np.array(self.table_dual).astype(float)
        
        print('\n\n------------------------------------------------')
        print('DUAL TABLE:\n\n', self.table_dual)

        r_index = DualOperators.get_pivot_row(self, a=self.table_dual)
        c_index = DualOperators.get_pivot_column(self, a=self.table_dual, row_index=r_index)
        self.table_dual = DualOperators.pivot_table(self, a=self.table_dual, row_index=r_index, col_index=c_index)
        print('------------------------------------------------')
        print(f'1º Iteração:\n')
        print(self.table_dual)
        index = 1
        
        while DualOperators.is_negative(self, a=self.table_dual):
            r_index = DualOperators.get_pivot_row(self, a=self.table_dual)
            c_index = DualOperators.get_pivot_column(self, a=self.table_dual, row_index=r_index)
            self.table_dual = DualOperators.pivot_table(self, a=self.table_dual, row_index=r_index, col_index=c_index)
            self.table_dual = np.round(self.table_dual, 2)
            print('\n------------------------------------------------')
            print(f'{index+1}º Iteração:\n')
            print('------------------------------------------------')
            print(self.table_dual)
            index += 1
            if index == 10:
                break
        
        DualOperators.get_results(self, a=self.table_dual)
        return self.table_dual

    def run_sensitivity_primal(self):
        for col in range(self.table_primal.shape[1]-1):
            if self.table_primal[:-1,col].max() == 1 and abs(self.table_primal[:-1,col]).sum() == 1:
                self.var_basic_index.append(col)
            else: 
                self.var_nonbasic_index.append(col)
        self.B_ = self.A_primal[:,self.var_basic_index]
        self.sensitivity_operations()
    
    def sensitivity_operations(self):
        b = self.b_primal[:-1]
        B_ = self.B_
        z = symbols('Z', extended_real=True)
        B_inv = Matrix(np.asarray(np.linalg.inv(np.matrix(B_))))
        variations = []
        for i in range(len(b)):
                
            b_2=Matrix(b)
            b_2[i]+=z
            eq = B_inv*b_2
            ineq =[]
            for linha in eq:
                ineq.append((linha,">"))
            ineq_ = []
            ineq_.append(ineq)
            result = reduce_rational_inequalities(ineq_, z)
            if result:
                print(f"Sensitivity Analysis x{self.var_basic_index[i]}: {result}")
        
            variations.append(result)

    def solve(self):
        self.primal()
        self.dual()
        self.run_primal()
        self.run_dual()
        self.run_sensitivity_primal()

if __name__ == '__main__':
    simplex = Simplex()

    ##Example MAX 01 - CORRETO {Z = 100.0, X_1 = 40.0, X_2 = 40.0}
    #simplex.objective_function('max', '1*x_1 + 1.5*x_2')
    #simplex.add_constraints(['2*x_1 + 2*x_2 <= 160', '1*x_1 + 2*x_2 <= 120', '4*x_1 + 2*x_2 <= 280'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0'])
    #simplex.solve()

    #Example MAX 02 - CORRETO {Z = 1.86, X_1 = 1.29, X_2 = 0.28}
    simplex.objective_function('max', '2*x_1 + 1*x_2')
    simplex.add_constraints(['3*x_1 + 4*x_2 <= 6', '6*x_1 + 1*x_2 <= 3'])
    simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0'])
    simplex.solve()

    #Example MAX 03 - CORRETO {Z = 43.95, X_1 = 11.98, X_2 = 16.05}
    #simplex.objective_function('max', '1*x_1 + 2*x_2')
    #simplex.add_constraints(['2*x_1 + 1*x_2 <= 40', '1*x_1 + 3*x_2 <= 60'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0'])
    #simplex.solve()

    #Example MAX 04 - CORRETO {Z = 36.0, X_1 = 2.0, X_2 = 6.0, X_3 = 2.0}
    #simplex.objective_function('max', '3*x_1 + 5*x_2')
    #simplex.add_constraints(['1*x_1 + 0*x_2 <= 4','0*x_1 + 2*x_2 <= 12','3*x_1 + 2*x_2 <= 18'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0'])
    #simplex.solve()

    #Example MAX 08 - CORRETO {Z = 2125.0, X_1 = 8.33, X_2 = 66.67, X_4 = 12.5}
    #simplex.objective_function('max', '10*x_1 + 20*x_2 + 30*x_3')
    #simplex.add_constraints(['2*x_1 + 2*x_2 + 4*x_3 <= 300','0*x_1 + 4*x_2 + 3*x_3 <= 200', '1*x_1 + 0*x_2 + 0*x_3 <= 20', '4*x_1 + 3*x_2 + 0*x_3 <= 50'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0', 'x_3 >=0'])
    #simplex.solve()

    #Example MIN 01 - CORRETO {Z = -60.0, X_2 = 10.0}
    #simplex.objective_function('min','3*x_1 + -6*x_2')
    #simplex.add_constraints(['4*x_1 + 2*x_2 <= 100','5*x_1 +7*x_2 <= 70'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0'])
    #simplex.solve()

    #Example MIN 02 - CORRETO {Z = -40.0, X_2 = 20.0}
    #simplex.objective_function('min','1*x_1 + -2*x_2')
    #simplex.add_constraints(['2*x_1 + 1*x_2 <= 40', '1*x_1 + 3*x_2 <= 60'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0'])
    #simplex.solve()

    #Example MIN 03 - | DUAS FASES | - CORRETO {Z = 5.28, X_1 = 7.5, X_2 = 0.31, X_3 = 4.55}
    #simplex.objective_function('min', '0.4*x_1 + 0.5*x_2')
    #simplex.add_constraints(['0.3*x_1 + 0.1*x_2 <= 2.7', '0.5*x_1 + 0.5*x_2 = 6', '0.6*x_1 + 0.4*x_2 >= 6'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0'])
    #simplex.solve()

    #Example MIN 04 |DUAS FASES| - CORRETO {Z = 36.0, X_1 = 1.0, X_2 = 1.5}
    #simplex.objective_function('min', '4*x_1 + 12*x_2 + 18*x_3')
    #simplex.add_constraints(['1*x_1 + 0*x_2 + 3*x_3 >= 3','0*x_1 + 2*x_2 + 2*x_3 >= 5'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0','x_3 >= 0'])
    #simplex.solve()

    #Example MAX 06 | DUAS FASES | - CORRETO {Z = -5.28, X_1 = 7.5 , X_2 = 0.31, X_3 = 4.55} 
    #simplex.objective_function('max', '-0.4*x_1 + -0.5*x_2')
    #simplex.add_constraints(['0.3*x_1 + 0.1*x_2 <= 2.7', '0.5*x_1 + 0.5*x_2 = 6', '0.6*x_1 + 0.4*x_2 >= 6'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0'])
    #simplex.solve()

    ###Example MAX 07 | DUAS FASE | - CORRETO {Z = 16.11, X_1 = 3.89, X_2 = 2.22}
    #simplex.objective_function('min', '3*x_1 + 2*x_2')
    #simplex.add_constraints(['2*x_1 + 1*x_2 >= 10', '1*x_1 + 5*x_2 >= 15'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0'])
    #simplex.solve()

    #Example MIN 04 | DUAS FASE | - CORRETO {Z = 132.2,X_2 = 2.7, X_4 = 0.69}
    #simplex.objective_function('min', '11*x_1 + 27*x_2 + 90*x_3')
    #simplex.add_constraints(['-1*x_1 + 1*x_2 + 2*x_3 >= 4','1*x_1 + 1*x_2 + 5*x_3 >= 6', '1*x_1 + 1*x_2 + 1*x_3 >= 4', '1*x_1 + 1*x_2 + 1*x_3 >= 1'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0','x_3 >= 0'])
    #simplex.solve()

    #Example MIN 05 | DUAS FASE | - CORRETO {Z = 7500.0, X_1 = 10.0, X_2 = 0.0}
    #simplex.objective_function('min', '1700*x_1 + 750*x_2 + 800*x_3')
    #simplex.add_constraints(['2*x_1 + 2*x_2 + 5*x_3 >= 20','3*x_1 + 1*x_2 + 5*x_3 = 10'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0','x_3 >= 0'])
    #simplex.solve()

    #Example MIN 05 | DUAS FASE | - CORRETO {Z = 7500.0, X_1 = 10.0, X_2 = 0.0}
    #simplex.objective_function('min', '1*x_1 + 2*x_2')
    #simplex.add_constraints(['-2*x_1 + 1*x_2 <= 3',
    #                        '3*x_1 + 4*x_2 <= 5',
    #                        '1*x_1 + -1*x_2 <= 2'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0'])
    #simplex.solve()

    #Example MIN 05 | DUAS FASE | - CORRETO {Z = 13.62, X_2 = 2.08, X_3 = 3.69}
    #simplex.objective_function('max', '1*x_1 + 2*x_2 + 3*x_3')
    #simplex.add_constraints(['1*x_1 + 1*x_2 + 1*x_3 <= 10','2*x_1 + 1*x_2 + 4*x_3 <= 12', '1*x_1 + 3*x_2 + -1*x_3 <= 9'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0','x_3 >= 0'])
    #simplex.solve()

    #Example MIN 06 | DUAS FASE | - CORRETO {Z = 21.0, X_1 = 4.0, X_2 = 1.0, X_3 = -21.0}
    #simplex.objective_function('min', '3*x_1 + 4*x_2 + 9*x_3')
    #simplex.add_constraints(['1*x_1 + 0*x_2 + 1*x_3 >= 5', '0*x_1 + 1*x_2 + 2*x_3 >= 2'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0','x_3 >= 0'])
    #simplex.solve()


    #Example MIN 08 | DUAS FASE | - CORRETO {Z = 19, X_1 = 4, X_2 = 5}
    #simplex.objective_function('max', '6*x_1 + -1*x_2')
    #simplex.add_constraints(['4*x_1 + 1*x_2 <= 21', '2*x_1 + 3*x_2 >= 13', '-1*x_1 + 1*x_2 = 1'])
    #simplex.add_var_constraints(['x_1 >= 0', 'x_2 >= 0'])
    #simplex.solve()

    #simplex.objective_function('max', '60*x_1 + 30*x_2 + 20*x_3')
    #simplex.add_constraints(['8*x_1 + 6*x_2 + 1*x_3 <= 48', '4*x_1 + 2*x_2 + 1.5*x_3 <= 20', '2*x_1+ 1.5*x_2 + 0.5*x_3 <= 8'])
    #simplex.add_var_constraints(['x_1 >= 0', 'x_2 >= 0', 'x_3 >= 0'])
    #simplex.solve()

    #Example MIN 04 |DUAS FASES| - CORRETO {Z = 36.0, X_1 = 1.0, X_2 = 1.5}
    #simplex.objective_function('max', '60*x_1 + 30*x_2 + 20*x_3 + 15*x_4')
    #simplex.add_constraints(['8*x_1 + 6*x_2 + 1*x_3 + 1*x_4 <= 48','4*x_1 + 2*x_2 + 1.5*x_3 + 1*x_4 <= 20', '2*x_1 + 1.5*x_2 + 0.5*x_3 + 1*x_4 <= 8'])
    #simplex.add_var_constraints(['x_1 >= 0','x_2 >= 0','x_3 >= 0', 'x_4 >= 0'])
    #simplex.solve()