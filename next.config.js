// next.config.js
module.exports = {
    webpack(config) {
      config.module.rules.push({
        test: /\.worker\.js$/,
        use: { loader: 'worker-loader' },
      });
      return config;
    },
  };
  