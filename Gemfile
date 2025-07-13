source "https://rubygems.org"

# Jekyll版本
gem "jekyll", "~> 4.3"

# 必要的依赖
gem "webrick", "~> 1.7"

# Jekyll插件
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
  gem "jekyll-sitemap"
  gem "jekyll-seo-tag"
end

# Windows系统支持
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", "~> 1.2"
  gem "tzinfo-data"
end

# 性能优化（Windows）
# 打开 Gemfile，找到 wdm 这行，改成：
#gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin], :require => false
