test_that(".calculate_content_loss calculates content loss", {
  content_data <- 1:5
  generated_data <- content_data
  generated_data[[1]] <- 2L
  content_layer <- torch::torch_tensor(
    data = content_data,
    dtype = torch::torch_float()
  )
  generated_layer <- torch::torch_tensor(
    data = generated_data,
    dtype = torch::torch_float()
  )
  content_loss <- .calculate_content_loss(content_layer, generated_layer)

  expect_s3_class(content_loss, "torch_tensor")
  expect_equal(
    as.numeric(content_loss),
    mean((content_data - generated_data)^2)
  )
})

test_that(".calculate_style_loss calculates style loss", {
  style_data <- array(1, dim = c(1, 2, 2, 2))
  generated_data <- style_data*2
  style_layer <- torch::torch_tensor(
    data = style_data,
    dtype = torch::torch_float()
  )
  generated_layer <- torch::torch_tensor(
    data = generated_data,
    dtype = torch::torch_float()
  )
  style_loss <- .calculate_style_loss(
    list(style_layer),
    list(generated_layer),
    1
  )

  expect_s3_class(style_loss, "torch_tensor")

  # These arrays get reshaped and self-multiplied, such that they effectively
  # lose a dimension. Let's set that up manually to make sure it comes back how
  # we assume.
  style_data_gram <- matrix(1, nrow = 2, ncol = 2)
  generated_data_gram <- matrix(2*2, nrow = 2, ncol = 2)
  simple_loss <- mean((style_data_gram - generated_data_gram)^2)
  expect_equal(
    as.numeric(style_loss),
    simple_loss
  )

  # Now let's repeat those to make sure weighting works.
  style_loss <- .calculate_style_loss(
    list(style_layer, style_layer),
    list(generated_layer, generated_layer),
    c(1, 0.5)
  )

  expect_s3_class(style_loss, "torch_tensor")
  expect_equal(
    as.numeric(style_loss),
    simple_loss * 1.5
  )

  # When they're identical, loss should be 0.
  generated_layer <- torch::torch_clone(style_layer)
  style_loss <- .calculate_style_loss(
    list(style_layer),
    list(generated_layer),
    1
  )

  expect_s3_class(style_loss, "torch_tensor")
  expect_equal(
    as.numeric(style_loss),
    0
  )
})
