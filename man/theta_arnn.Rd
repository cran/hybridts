% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/hybrid_ts.R
\name{theta_arnn}
\alias{theta_arnn}
\title{Hybrid Theta ARNN Forecasting Model}
\usage{
theta_arnn(y, n, PI = FALSE, ret_fit = FALSE)
}
\arguments{
\item{y}{A numeric vector or time series}

\item{n}{An integer specifying the forecast horizon}

\item{PI}{A logical flag (default = \code{FALSE}) for generating the prediction interval.}

\item{ret_fit}{A logical flag specifying that the fitted values of the model on the
training set should be returned if true, otherwise, false (default)}
}
\value{
The forecast of the time series of size \code{n} is generated along with the optional
output of fitted values (\code{ret_fit} = TRUE) and confidence interval (\code{PI} = TRUE) for the forecast.
}
\description{
Hybrid Theta ARNN Forecasting Model
}
\examples{
theta_arnn(y = datasets::lynx, n = 3)

}
\references{
\itemize{
\item Bhattacharyya, A., Chakraborty, T., & Rai, S. N. (2022). Stochastic forecasting of
COVID-19 daily new cases across countries with a novel hybrid time series model.
Nonlinear Dynamics, 1-16.

\item Bhattacharyya, A., Chattopadhyay, S., Pattnaik, M., & Chakraborty, T. (2021, July).
Theta Autoregressive Neural Network: A Hybrid Time Series Model for Pandemic Forecasting.
In 2021 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.}
}
