import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime

class stress_testing:
    '''
    tikers: a list with ticker names
    positions: a numpy array with weights in each stock considered. The weights has to sum to 1.
    n_sims: int, that determines the amount of simulations.
    start_data and end_date: datetime objects

    The object conntains two methods.
    1: montecarlo_sim.
    Simulates a correlated returns time series of the data.
    2: asses_worst_cases.
    Estimates the lower quantile of the simulated paths, and estimates the worst cases of the portfolio.

    '''

    def __init__(self, tickers, positions, n_sims = 1000,
                 start_date = (datetime.datetime.now() - datetime.timedelta(days=3*365)).date(),
                 end_date = datetime.datetime.now().date()):

        if len(tickers) != len(positions):
            raise Exception('The dimension of the position is not same as the chosen stocks')

        self.tickers = tickers
        self.positions = positions
        if sum(self.positions) != 1:
            raise Exception('The weights does not sum up to 1')

        self.start_date = start_date
        self.end_date = end_date
        self.data = self.get_data()
        self.n_sims = n_sims
        self.returns = self.data.pct_change()
        self.portfolio_values = np.zeros((len(self.returns), self.n_sims))
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()

    def get_data(self):
        m = ' '.join(self.tickers)
        data = yf.download(
            m,
            start = self.start_date,
            end = self.end_date
        )['Adj Close']
        return data

    def monte_carlo_sim(self):
        for i in range(self.n_sims):
            scenario_returns = np.random.multivariate_normal(self.mean_returns, self.cov_matrix, len(self.returns))
            scenario_returns = pd.DataFrame(scenario_returns, columns=self.returns.columns)
            # Calculate the value for each scenario
            cop = self.portfolio_values

            scenario_portfolio = (1+scenario_returns).cumprod()
            self.portfolio_values[:, i] = np.dot(
                self.positions, scenario_portfolio.T
            )

        for i in range(self.n_sims):
            plt.plot(self.portfolio_values[:, i], linewidth=0.5, alpha=0.3)

        plt.title('Monte Carlo Simulation - Portfolio Stress Testing')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        # plt.plot(np.dot(self.positions, (1+self.returns.cumprod()).T))
        c_r = (1+self.returns).cumprod()
        pf_val = np.dot(self.positions, c_r.T)
        plt.plot(pf_val, label = 'Actual portfolio performance')
        plt.legend(loc = 'upper left')
        plt.show()

    def asses_worst_cases(self):

        q = np.quantile(self.portfolio_values[-1: ][0], 0.025)
        plt.hist(np.array(self.portfolio_values[-1: ][0]), bins = 20, density=True)
        plt.axvline(q, color = 'red', label = '2.5 percent quantile ')
        plt.legend()
        plt.show()

        print(f'In 97.5 percent of the cases, the portfolio does not lose more than {round(1-q, 4)} percent of its value'
              )


### Example of usage ###

# Define the input for the object.
start_date = datetime.date(2022, 11, 1)
end_date = datetime.date(2023, 11, 1)
ticker_symbols = ['META', 'MSFT', 'GOOGL']
weights = np.array([0.4, 0.5, 0.1])

# Initiate the object.

obj1 = stress_testing(['META', 'MSFT', 'GOOGL'], weights, n_sims=10, start_date=start_date, end_date=end_date)

# Apply the monte carlo simulation scheme.

obj1.monte_carlo_sim()
obj1.asses_worst_cases()
