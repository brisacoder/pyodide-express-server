# Contributing

Thanks for your interest in improving the Pyodide Express Server! All types of contributions are welcome, especially those related to our **enhanced security logging system** and **interactive dashboard**.

## 🚀 Getting Started

### Prerequisites
- **Node.js** (v18 or later) and **npm** (v8 or later)
- **Python** (v3.9 or later) for running the test suite
- **Git** for version control

### Setup
1. **Fork the repository** and create your branch from `main`.
2. **Clone your fork** and navigate to the project directory.
3. **Install Node.js dependencies**: `npm ci`
4. **Set up Python testing environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```
5. **Copy environment template**: `cp .env.example .env` and adjust values as needed.
6. **Start the development server**: `npm run dev`

## 🔧 Development Workflow

### Code Quality
- **Linting**: Use `npm run lint` to lint the code
- **Formatting**: Use `npm run format` to apply Prettier formatting
- **Type Checking**: Ensure TypeScript definitions are correct (if applicable)

### Testing Strategy
We have comprehensive testing across multiple dimensions:

#### Quick Development Testing (39 tests ~1 second)
```bash
# Start server first
npm run dev

# In another terminal, run quick tests
python run_simple_tests.py
```

#### Comprehensive Testing (100+ tests ~60 seconds)
```bash
# Automatic server management - no manual setup needed
python run_comprehensive_tests.py

# Run specific categories
python run_comprehensive_tests.py --categories basic security_logging

# Run security-focused tests
python run_comprehensive_tests.py --categories security security_logging
```

#### Security Logging Tests (New!)
```bash
# Run just the enhanced security logging tests
python -m unittest tests.test_security_logging -v

# Test specific security features
python -m unittest tests.test_security_logging.SecurityLoggingTestCase.test_06_dashboard_endpoints_functionality -v
```

### 🔐 Security-Related Contributions

When contributing to security features:

1. **Maintain Backward Compatibility**: All security enhancements must not break existing APIs
2. **Add Comprehensive Tests**: Security features require extensive test coverage
3. **Update Documentation**: Include security implications in documentation
4. **Test Dashboard Integration**: Verify interactive dashboard continues to work
5. **Verify Audit Trails**: Ensure security logging captures relevant events

#### Security Testing Checklist
- [ ] All existing tests still pass (backward compatibility)
- [ ] New security tests added and passing
- [ ] Dashboard endpoints function correctly
- [ ] Security logging produces valid audit trails
- [ ] Performance impact is minimal (<5% overhead)

## 📊 Dashboard & UI Contributions

For dashboard and visualization improvements:

1. **Use Chart.js**: Maintain consistency with existing visualizations
2. **Responsive Design**: Ensure compatibility across devices
3. **Professional Styling**: Follow existing CSS patterns and gradients
4. **Real-time Updates**: Consider WebSocket integration for live updates
5. **Accessibility**: Include ARIA labels and keyboard navigation

## 🧪 Adding New Tests

### Test Categories
- **Basic API**: Core endpoint functionality
- **Security**: Input validation and authentication
- **Security Logging**: Enhanced monitoring and audit trails (New!)
- **Performance**: Load and stress testing
- **Integration**: End-to-end workflows
- **Data Science**: Matplotlib, Seaborn, Scikit-learn functionality

### Test Guidelines
1. **Descriptive Names**: Test names should clearly describe what's being tested
2. **Setup/Teardown**: Clean up after tests (files, statistics, etc.)
3. **Independence**: Tests should not depend on each other
4. **Error Cases**: Test both success and failure scenarios
5. **Performance**: Include timing assertions where relevant

### Adding to Test Runners
When adding new test modules:
1. Add to `test_modules` list in `run_comprehensive_tests.py`
2. Add category mapping in the comprehensive runner
3. Update `run_simple_tests.py` if it's a core feature
4. Update `TESTING.md` documentation

## 📝 Code Style Guidelines

### JavaScript/Node.js
- **ES6+ Features**: Use modern JavaScript (async/await, destructuring, etc.)
- **Error Handling**: Always include proper try/catch blocks
- **Comments**: Use JSDoc for complex functions
- **Security**: Validate all user inputs

### Python (Test Code)
- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Include type hints where helpful
- **Docstrings**: Document test methods clearly
- **Assertions**: Use descriptive assertion messages

### Documentation
- **Markdown**: Use proper markdown formatting
- **Code Blocks**: Include language identifiers
- **Examples**: Provide working examples
- **Links**: Use relative links for internal references

## 🛡️ Security Considerations

All contributions should consider:

1. **Input Validation**: Sanitize and validate all user inputs
2. **Rate Limiting**: Consider impact on server resources
3. **Audit Logging**: Ensure security events are properly logged
4. **Error Handling**: Don't leak sensitive information in error messages
5. **Dependencies**: Keep dependencies up to date and scan for vulnerabilities

## 📋 Pull Request Process

1. **Create Feature Branch**: Branch from `main` with descriptive name
2. **Develop & Test**: Implement changes with comprehensive testing
3. **Update Documentation**: Update relevant `.md` files
4. **Run Full Test Suite**: Ensure all tests pass
5. **Clear Commit Messages**: Use conventional commit format
6. **Open Pull Request**: Include detailed description of changes

### PR Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review of the code completed
- [ ] Added/updated tests for new functionality
- [ ] All tests pass (run `python run_comprehensive_tests.py`)
- [ ] Documentation updated (README, API docs, etc.)
- [ ] Security implications considered and tested
- [ ] Backward compatibility maintained

## 🚨 Reporting Issues

### Bug Reports
When reporting bugs, include:
- **Environment**: Node.js version, OS, Python version
- **Steps to Reproduce**: Detailed reproduction steps
- **Expected vs Actual**: What you expected vs what happened
- **Logs**: Relevant error messages and log entries
- **Test Results**: Output from `run_comprehensive_tests.py`

### Security Issues
For security-related issues:
- **Private Disclosure**: Contact maintainers privately first
- **Detailed Description**: Include potential impact assessment
- **Proof of Concept**: Provide minimal reproduction steps
- **Suggested Fix**: If you have ideas for fixes

### Feature Requests
For new feature requests:
- **Use Case**: Describe the problem you're trying to solve
- **Proposed Solution**: Your preferred implementation approach
- **Alternatives**: Other approaches you've considered
- **Security Impact**: Any security implications
- **Breaking Changes**: Whether it would break existing functionality

## 🏗️ Architecture Guidelines

When contributing architectural changes:

1. **Modular Design**: Keep components loosely coupled
2. **Configuration**: Use environment variables for configurable values
3. **Error Boundaries**: Implement proper error handling at service boundaries
4. **Logging**: Include appropriate logging at different levels
5. **Testing**: Design with testability in mind

## 🎯 Development Priorities

Current focus areas for contributions:

### High Priority
- **Security Enhancements**: Rate limiting, authentication, advanced threat detection
- **Performance Optimization**: Caching, connection pooling, resource management
- **Dashboard Improvements**: Real-time updates, custom visualizations, export features

### Medium Priority
- **Documentation**: Video tutorials, interactive examples, troubleshooting guides
- **Developer Tools**: VS Code extensions, CLI tools, SDKs
- **Integration**: External monitoring tools, CI/CD improvements

### Research Areas
- **Machine Learning Integration**: Model training and deployment
- **Edge Computing**: Offline capabilities, mobile integration
- **Advanced Security**: Zero-trust architecture, homomorphic encryption

## 🤝 Community Guidelines

- **Be Respectful**: Treat all contributors with respect and kindness
- **Be Patient**: Review feedback constructively and respond thoughtfully
- **Be Collaborative**: Work together to find the best solutions
- **Be Learning-Oriented**: Share knowledge and learn from others

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Documentation**: Check all `.md` files for detailed information
- **Test Suite**: Run tests to understand expected behavior

Thank you for contributing to the Pyodide Express Server! Your contributions help make this project better for everyone. 🎉
